"""用 langgraph 编排的问答 chat：LLM 先改写查询，再用 bm25s 检索，最后据资料作答。

图结构：START -> expand -> retrieve -> chatbot -> END
expand   节点用 LLM 把问题改写成数条检索查询（中文问句会额外给出英文查询）
retrieve 节点对每条查询跑 bm25s，融合去重后用重排模型按原始问题排序，取 top-k 写入 context
chatbot  节点据 context 与对话历史作答

用法:
    uv run python -m taste_rag.bm25_chat
"""

import os
import re
import warnings
from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, SystemMessage
from langchain_core._api import LangChainBetaWarning
from langchain_core.documents import Document
from langchain_reranker.rerankers import init_reranker
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, MessagesState, StateGraph

from taste_rag.bm25 import build_index, load_chunks, search

DB_PATH = Path(__file__).resolve().parents[2] / ".bm25_chat.db"
THREAD_ID = "bm25-chat-test"
N_QUERIES = 3
RECALL_K = 30  # 每条查询的召回条数（重排前先放宽）
TOP_K = 5  # 重排后进入上下文的条数
SYSTEM_PROMPT = "你是客服助手，只依据给定资料回答问题；资料中没有的信息，直接说明不知道。"
QUERY_PROMPT = """把下面的问题改写成 {n} 条用于关键词检索的查询，每行一条，不要编号。
要求：覆盖不同的关键词说法；如果参考资料是英文，请给出英文查询。

问题：{question}"""
LIST_PREFIX = re.compile(r"^\s*(?:[-*•]|\d+[.、)])\s*")


class ChatState(MessagesState):
    """messages 是对话；queries 是本轮改写出的检索查询；context 是检索到的资料。"""

    queries: list[str]
    context: str


def expand_queries(model, question: str, n: int = N_QUERIES) -> list[str]:
    """让 LLM 把问题改写成 n 条检索查询；失败则退回原问题。"""
    raw = model.invoke(
        [HumanMessage(content=QUERY_PROMPT.format(n=n, question=question))]
    ).text
    queries = [LIST_PREFIX.sub("", line).strip() for line in raw.splitlines()]
    queries = [query for query in queries if query][:n]
    return queries or [question]


def build_graph(model, retriever, chunks, reranker, checkpointer=None):
    """构建并编译 chat 图。"""

    def expand(state: ChatState) -> dict:
        queries = expand_queries(model, state["messages"][-1].text)
        print(f"[自动查询] {' | '.join(queries)}")
        return {"queries": queries}

    def retrieve(state: ChatState) -> dict:
        # 1) 每条查询各召回 RECALL_K 条，按累加得分融合，同一分块只保留一次
        merged: dict[tuple[str, str], float] = {}
        for query in state["queries"]:
            for score, source, text in search(retriever, chunks, query, RECALL_K):
                key = (source, text)
                merged[key] = merged.get(key, 0.0) + score
        candidates = [
            Document(page_content=text, metadata={"source": source})
            for (source, text), _ in sorted(
                merged.items(), key=lambda item: item[1], reverse=True
            )
        ]
        # 2) 用重排模型按“原始问题”排序（重排是多语言的，中文问题可直接给英文片段打分）
        ranked = reranker.compress_documents(
            candidates, query=state["messages"][-1].text
        )[:TOP_K]
        return {
            "context": "\n\n".join(
                f"[{doc.metadata['source']}] {doc.page_content}" for doc in ranked
            )
        }

    def chatbot(state: ChatState) -> dict:
        prompt = f"{SYSTEM_PROMPT}\n\n资料：\n{state['context']}"
        reply = model.invoke([SystemMessage(content=prompt), *state["messages"]])
        return {"messages": [reply]}

    builder = StateGraph(ChatState)
    builder.add_node("expand", expand)
    builder.add_node("retrieve", retrieve)
    builder.add_node("chatbot", chatbot)
    builder.add_edge(START, "expand")
    builder.add_edge("expand", "retrieve")
    builder.add_edge("retrieve", "chatbot")
    builder.add_edge("chatbot", END)
    return builder.compile(checkpointer=checkpointer)


def main() -> None:
    # v3 流式协议目前是实验性的，屏蔽其 beta 警告避免污染 CLI 输出
    warnings.filterwarnings("ignore", category=LangChainBetaWarning)

    load_dotenv()
    chunks = load_chunks()
    retriever = build_index(chunks)
    print(f"已索引 {len(chunks)} 个分块")

    reranker = init_reranker(
        os.environ["RERANKER_MODEL"], base_url=os.environ["RERANKER_BASE_URL"]
    )

    with SqliteSaver.from_conn_string(str(DB_PATH)) as checkpointer:
        graph = build_graph(
            init_chat_model(os.environ["MODEL"]),
            retriever,
            chunks,
            reranker,
            checkpointer,
        )
        config = {"configurable": {"thread_id": THREAD_ID}}

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
            # expand 节点会先打印改写结果，故 "A: " 延迟到首个回答 token 再输出
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
