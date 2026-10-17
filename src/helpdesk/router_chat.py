"""并行检索 + Router 决定是否检索的问答 chat。

图结构：START -> router
         router --检索--> bm25(40) ─┐
         router --检索--> semantic(20) ─> combine(去重) -> rerank(5) -> chatbot
         router --不检索--------------------> chatbot
         chatbot -> END

Router 支持两种模式（ROUTER_MODE）：
  - rule（默认）：0ms 规则，寒暄/致谢判「不检索」，其余默认「检索」。
    实测当前 2.6B 模型关掉思考后无法做这个判断（一律答 n），开思考又要 1.5~3s，故默认走规则。
  - llm：用 with_structured_output 让模型直接返回目标节点名（retrieve / chatbot），
    省掉文本解析；temperature=0，带上最近几轮历史（截断控制），
    历史已能回答的问题不再检索；异常时保守地默认「检索」。
    配 ROUTER_MODEL / ROUTER_BASE_URL 指向非思考小模型。

用法:
    uv run python src/helpdesk/router_chat.py
"""

import os
import re
import time
import warnings
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, SystemMessage
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_reranker.rerankers import init_reranker
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, MessagesState, StateGraph
from pydantic import BaseModel, Field

from helpdesk import DEFAULT_THREAD, DEFAULT_USER, silence_beta_warning, thread_config
from helpdesk.bm25 import build_index, load_chunks, search
from helpdesk.rag import COLLECTION, INDEX_DIR

DB_PATH = Path(__file__).resolve().parents[2] / ".router_chat.db"
BM25_K = 40
SEMANTIC_K = 20
TOP_K = 5
ROUTER_MAX_TOKENS = 64  # 结构化输出只需几个 token，留点余量
HISTORY_MESSAGES = 4  # 路由参考的历史条数（不含当前问题）
HISTORY_CHARS = 200  # 每条历史最多保留的字符数，控制 prefill 成本

# Router：由 schema 约束输出目标节点名，无需解析文本。
# 注意别问「是否需要检索外部资料」——模型会因"这我懂"一律答否；问「属于哪一类」才准。
# 历史已能回答的问题（追问、指代）走 chatbot，避免多余检索。
ROUTER_PROMPT = (
    "你在 LangChain/LangGraph 技术客服场景工作。结合对话历史判断当前问题：\n"
    "- 历史里已有足够信息可以直接回答 -> chatbot\n"
    "- 寒暄、闲聊、或通用常识，不需要查文档 -> chatbot\n"
    "- 需要查阅 LangChain/LangGraph 技术文档才能准确回答 -> retrieve\n"
    "拿不准选 retrieve。\n\n"
    "[对话历史]\n{history}\n\n"
    "[当前问题]\n{question}"
)

# 0ms 快路径：纯寒暄/确认显然不需要查库，直接跳过 LLM 路由
SMALL_TALK = re.compile(
    r"^\s*(你好|您好|hi|hello|hey|谢谢|感谢|多谢|thanks|thank you|ok|okay|好的|收到|再见|bye)"
    r"[\s!！。.？?]*$",
    re.IGNORECASE,
)

SYSTEM_PROMPT = "你是客服助手，只依据给定资料回答问题；资料中没有的信息，直接说明不知道。"
PLAIN_PROMPT = "你是客服助手，请直接、简洁地回答用户。"


class RouteDecision(BaseModel):
    """Router 的输出：直接给出目标节点名，省掉文本解析。"""

    next: Literal["retrieve", "chatbot"] = Field(
        description="retrieve=需要检索知识库；chatbot=直接回答"
    )


class ChatState(MessagesState):
    """messages 是对话；search 是路由结果；其余是各阶段产物。"""

    search: bool
    bm25_docs: list[dict]
    semantic_docs: list[dict]
    candidates: list[dict]
    context: str


def silence_serializer_warning() -> None:
    """屏蔽结构化输出带来的噪声。

    with_structured_output 把解析结果放进消息的 parsed 字段，LangGraph 在
    checkpoint/事件序列化时会报 PydanticSerializationUnexpectedValue。
    解析结果本身是正确的，只是序列化器不认识这个类型。
    """
    warnings.filterwarnings("ignore", message="Pydantic serializer warnings")


def rule_needs_retrieval(question: str) -> bool:
    """0ms 规则判定：寒暄/致谢不需要查库，其余默认检索（宁可多查）。"""
    return not SMALL_TALK.match(question)


def format_history(messages, max_messages: int = HISTORY_MESSAGES) -> str:
    """取最近若干条历史（不含当前问题），截断后拼成路由可读的文本。"""
    lines = []
    for message in messages[-(max_messages + 1) : -1]:
        text = " ".join(str(message.text).split())[:HISTORY_CHARS]
        if text:
            lines.append(f"{message.type}: {text}")
    return "\n".join(lines) or "（无历史）"


def needs_retrieval(structured_router, question: str, history: str) -> bool:
    """LLM 路由：结合历史判断，schema 直接返回目标节点名；异常时保守返回 True（检索）。"""
    try:
        route = structured_router.invoke(
            [
                HumanMessage(
                    content=ROUTER_PROMPT.format(history=history, question=question)
                )
            ]
        )
    except Exception:  # noqa: BLE001 - 路由失败不应阻塞对话
        return True
    return route.next == "retrieve"


def build_graph(
    model,
    router_model,
    bm25,
    chunks,
    retriever,
    reranker,
    checkpointer=None,
    router_mode: str = "rule",
):
    """构建并编译 chat 图。"""
    # 结构化输出：模型直接返回目标节点名
    structured_router = router_model.with_structured_output(RouteDecision)

    def router(state: ChatState) -> dict:
        messages = state["messages"]
        question = messages[-1].text
        started = time.perf_counter()
        if router_mode == "llm":
            search_needed = needs_retrieval(
                structured_router, question, format_history(messages)
            )
        else:
            search_needed = rule_needs_retrieval(question)
        cost = time.perf_counter() - started
        print(
            f"[router] {'检索' if search_needed else '不检索'}"
            f"  {cost * 1000:.0f}ms  ({router_mode})"
        )
        # 每轮先清空 context，避免上一轮的检索结果被沿用
        return {"search": search_needed, "context": ""}

    def bm25_retrieve(state: ChatState) -> dict:
        question = state["messages"][-1].text
        hits = search(bm25, chunks, question, BM25_K)
        return {"bm25_docs": [{"source": s, "text": t} for _, s, t in hits]}

    def semantic_retrieve(state: ChatState) -> dict:
        question = state["messages"][-1].text
        return {
            "semantic_docs": [
                {"source": doc.metadata["source"], "text": doc.page_content}
                for doc in retriever.invoke(question)
            ]
        }

    def combine(state: ChatState) -> dict:
        merged: list[dict] = []
        seen: set[tuple[str, str]] = set()
        for doc in state["bm25_docs"] + state["semantic_docs"]:
            key = (doc["source"], doc["text"])
            if key not in seen:
                seen.add(key)
                merged.append(doc)
        return {"candidates": merged}

    def rerank(state: ChatState) -> dict:
        documents = [
            Document(page_content=c["text"], metadata={"source": c["source"]})
            for c in state["candidates"]
        ]
        ranked = reranker.compress_documents(
            documents, query=state["messages"][-1].text
        )[:TOP_K]
        return {
            "context": "\n\n".join(
                f"[{doc.metadata['source']}] {doc.page_content}" for doc in ranked
            )
        }

    def chatbot(state: ChatState) -> dict:
        prompt = (
            f"{SYSTEM_PROMPT}\n\n资料：\n{state['context']}"
            if state["context"]
            else PLAIN_PROMPT
        )
        reply = model.invoke([SystemMessage(content=prompt), *state["messages"]])
        return {"messages": [reply]}

    builder = StateGraph(ChatState)
    builder.add_node("router", router)
    builder.add_node("bm25", bm25_retrieve)
    builder.add_node("semantic", semantic_retrieve)
    builder.add_node("combine", combine)
    builder.add_node("rerank", rerank)
    builder.add_node("chatbot", chatbot)

    builder.add_edge(START, "router")
    builder.add_conditional_edges(
        "router",
        lambda state: ["bm25", "semantic"] if state["search"] else "chatbot",
    )
    builder.add_edge("bm25", "combine")
    builder.add_edge("semantic", "combine")
    builder.add_edge("combine", "rerank")
    builder.add_edge("rerank", "chatbot")
    builder.add_edge("chatbot", END)
    return builder.compile(checkpointer=checkpointer)


def main() -> None:
    silence_beta_warning()
    silence_serializer_warning()
    load_dotenv()

    chunks = load_chunks()
    bm25 = build_index(chunks)
    embeddings = OpenAIEmbeddings(
        model=os.environ["EMBEDDING_MODEL"],
        base_url=os.environ["EMBEDDING_BASE_URL"],
        api_key=os.environ["OPENAI_API_KEY"],
        check_embedding_ctx_length=False,
    )
    retriever = Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings,
        persist_directory=str(INDEX_DIR),
    ).as_retriever(search_kwargs={"k": SEMANTIC_K})
    reranker = init_reranker(
        os.environ["RERANKER_MODEL"], base_url=os.environ["RERANKER_BASE_URL"]
    )
    print(f"已索引 {len(chunks)} 个分块")

    model = init_chat_model(os.environ["MODEL"])
    # Router 可以指向另一个服务（例如非思考的小模型），默认复用对话模型
    router_model = init_chat_model(
        os.environ.get("ROUTER_MODEL", os.environ["MODEL"]),
        base_url=os.environ.get("ROUTER_BASE_URL", os.environ["OPENAI_API_BASE"]),
        api_key=os.environ["OPENAI_API_KEY"],
        temperature=0,
        max_tokens=ROUTER_MAX_TOKENS,
    )
    router_mode = os.environ.get("ROUTER_MODE", "rule")
    config = thread_config(DEFAULT_USER, DEFAULT_THREAD)

    with SqliteSaver.from_conn_string(str(DB_PATH)) as checkpointer:
        graph = build_graph(
            model,
            router_model,
            bm25,
            chunks,
            retriever,
            reranker,
            checkpointer,
            router_mode,
        )

        print("输入问题开始问答，输入 exit / quit 结束。")
        while True:
            try:
                question = input("Q: ")
            except (EOFError, KeyboardInterrupt):
                break
            if question.strip().lower() in {"exit", "quit"}:
                break

            stream = graph.stream_events(
                {"messages": [HumanMessage(content=question)]},
                config,
                version="v3",
            )
            # router 会先打印决策，"A: " 延迟到首个回答 token 再输出
            started = False
            for _, item in stream.interleave("messages"):
                if item.node != "chatbot":
                    continue
                for token in item.text:
                    if not started:
                        print("A: ", end="", flush=True)
                        started = True
                    print(token, end="", flush=True)
            print()


if __name__ == "__main__":
    main()
