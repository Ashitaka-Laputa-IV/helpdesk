"""查询改写 + bm25 多路召回 + 重排的问答 chat。

图结构：START -> expand -> retrieve -> chatbot -> END
expand   节点用 LLM 把问题改写成数条检索查询（中文问句会额外给出英文查询）
retrieve 节点对每条查询跑 bm25s，按累加得分融合后用重排模型排序，取 top-k 写入 context
chatbot  节点据 context 与对话历史作答

用法:
    uv run python -m taste_rag.rag.bm25_rerank
"""

import re
from pathlib import Path

from dotenv import load_dotenv
from langchain.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from taste_rag.core import runtime
from taste_rag.core.bm25 import search
from taste_rag.rag.common import RagState, make_answer_node
from taste_rag.core.retrieval import format_context, rerank_candidates

DB_PATH = Path(__file__).resolve().parents[2] / ".bm25_chat.db"
THREAD_ID = "bm25-chat-test"
N_QUERIES = 3
RECALL_K = 30  # 每条查询的召回条数（重排前先放宽）
TOP_K = 5  # 重排后进入上下文的条数
QUERY_PROMPT = """把下面的问题改写成 {n} 条用于关键词检索的查询，每行一条，不要编号。
要求：覆盖不同的关键词说法；如果参考资料是英文，请给出英文查询。

问题：{question}"""
LIST_PREFIX = re.compile(r"^\s*(?:[-*•]|\d+[.、)])\s*")


class MultiQueryState(RagState):
    """在 RagState 之上记录本轮改写出的检索查询。"""

    queries: list[str]


def expand_queries(model, question: str, n: int = N_QUERIES) -> list[str]:
    """让 LLM 把问题改写成 n 条检索查询；失败则退回原问题。"""
    raw = model.invoke(
        [HumanMessage(content=QUERY_PROMPT.format(n=n, question=question))]
    ).text
    queries = [LIST_PREFIX.sub("", line).strip() for line in raw.splitlines()]
    queries = [query for query in queries if query][:n]
    return queries or [question]


def build_graph(model, bm25, chunks, reranker, checkpointer=None):
    """构建并编译 chat 图。"""

    def expand(state: MultiQueryState) -> dict:
        queries = expand_queries(model, state["messages"][-1].text)
        print(f"[自动查询] {' | '.join(queries)}")
        return {"queries": queries}

    def retrieve(state: MultiQueryState) -> dict:
        # 1) 每条查询各召回 RECALL_K 条，按累加得分融合，同一分块只保留一次
        merged: dict[tuple[str, str], float] = {}
        for query in state["queries"]:
            for score, source, text in search(bm25, chunks, query, RECALL_K):
                key = (source, text)
                merged[key] = merged.get(key, 0.0) + score
        candidates = [
            {"source": source, "text": text}
            for (source, text), _ in sorted(
                merged.items(), key=lambda item: item[1], reverse=True
            )
        ]
        # 2) 用重排模型按“原始问题”排序（重排是多语言的，中文问题可直接给英文片段打分）
        top = rerank_candidates(
            reranker, candidates, state["messages"][-1].text, TOP_K
        )
        return {"context": format_context(top)}

    builder = StateGraph(MultiQueryState)
    builder.add_node("expand", expand)
    builder.add_node("retrieve", retrieve)
    builder.add_node("chatbot", make_answer_node(model))
    builder.add_edge(START, "expand")
    builder.add_edge("expand", "retrieve")
    builder.add_edge("retrieve", "chatbot")
    builder.add_edge("chatbot", END)
    return builder.compile(checkpointer=checkpointer)


def main() -> None:
    runtime.silence_warnings()
    load_dotenv()

    bm25, chunks = runtime.build_bm25()
    with SqliteSaver.from_conn_string(str(DB_PATH)) as checkpointer:
        graph = build_graph(
            runtime.load_model(), bm25, chunks, runtime.load_reranker(), checkpointer
        )
        config = {"configurable": {"thread_id": THREAD_ID}}
        runtime.run_cli(lambda question: runtime.stream_answer(graph, config, question))


if __name__ == "__main__":
    main()
