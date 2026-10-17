"""并行检索 + Router 决定是否检索的问答 chat。

图结构：START -> router
         router --检索--> bm25(40) ─┐
         router --检索--> semantic(20) ─> combine(去重) -> rerank(5) -> chatbot
         router --不检索--------------------> chatbot
         chatbot -> END

Router 分两级降低时延：
  1) 0ms 规则快路径：纯寒暄/确认直接判为「不检索」，不进模型；
  2) 其余交给 LLM：最短提示（只答 y/n）+ temperature=0 + reasoning_effort=none，
     只喂当前问题、不带历史；异常或输出不可解析时保守地默认「检索」。

用法:
    uv run python src/helpdesk/router_chat.py
"""

import os
import re
import time
import warnings
from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, SystemMessage
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_reranker.rerankers import init_reranker
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, MessagesState, StateGraph

from helpdesk import DEFAULT_THREAD, DEFAULT_USER, silence_beta_warning, thread_config
from helpdesk.bm25 import build_index, load_chunks, search
from helpdesk.rag import COLLECTION, INDEX_DIR

DB_PATH = Path(__file__).resolve().parents[2] / ".router_chat.db"
BM25_K = 40
SEMANTIC_K = 20
TOP_K = 5
ROUTER_MAX_TOKENS = 256  # 只防跑飞；当前模型思考要 100+ token，卡太小会得到空串

# Router：不带历史、不解释、只答一个字母。
# 实测（LFM2.5-2.6B）：y/n 版比 yes/no 版快一倍；再配 reasoning_effort="none" 可由 ~3.5s 降到 ~1.4s。
ROUTER_PROMPT = "只输出 y 或 n。是否需要检索外部资料来回答「{question}」？"

# 0ms 快路径：纯寒暄/确认显然不需要查库，直接跳过 LLM 路由
SMALL_TALK = re.compile(
    r"^\s*(你好|您好|hi|hello|hey|谢谢|感谢|多谢|thanks|thank you|ok|okay|好的|收到|再见|bye)"
    r"[\s!！。.？?]*$",
    re.IGNORECASE,
)

SYSTEM_PROMPT = "你是客服助手，只依据给定资料回答问题；资料中没有的信息，直接说明不知道。"
PLAIN_PROMPT = "你是客服助手，请直接、简洁地回答用户。"


class ChatState(MessagesState):
    """messages 是对话；search 是路由结果；其余是各阶段产物。"""

    search: bool
    bm25_docs: list[dict]
    semantic_docs: list[dict]
    candidates: list[dict]
    context: str


def needs_retrieval(router_model, question: str) -> bool:
    """判断是否需要检索：先走 0ms 规则，再问 LLM；异常或不可解析时保守返回 True。"""
    if SMALL_TALK.match(question):
        return False
    try:
        raw = router_model.invoke(
            [HumanMessage(content=ROUTER_PROMPT.format(question=question))]
        ).text
    except Exception:  # noqa: BLE001 - 路由失败不应阻塞对话
        return True
    letters = "".join(ch for ch in raw.lower() if ch.isalpha())
    return not letters.startswith("n")


def build_graph(model, router_model, bm25, chunks, retriever, reranker, checkpointer=None):
    """构建并编译 chat 图。"""
    # 不采样；尽量抑制思考链（实测可把路由耗时压掉一半以上）
    fast_router = router_model.bind(
        temperature=0,
        reasoning_effort="none",
        max_tokens=ROUTER_MAX_TOKENS,
    )

    def router(state: ChatState) -> dict:
        question = state["messages"][-1].text
        started = time.perf_counter()
        search_needed = needs_retrieval(fast_router, question)
        cost = time.perf_counter() - started
        print(f"[router] {'检索' if search_needed else '不检索'}  {cost * 1000:.0f}ms")
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
    router_model = init_chat_model(os.environ.get("ROUTER_MODEL", os.environ["MODEL"]))
    config = thread_config(DEFAULT_USER, DEFAULT_THREAD)

    with SqliteSaver.from_conn_string(str(DB_PATH)) as checkpointer:
        graph = build_graph(model, router_model, bm25, chunks, retriever, reranker, checkpointer)

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
