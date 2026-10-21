"""Router 决定是否检索的问答 chat，检索后排序方式可切换。

图结构：START -> summarize -> router
         router --检索--> bm25 ─┐
         router --检索--> 向量 ─> rank -> chatbot
         router --不检索---------------> chatbot
         chatbot -> END

rank 节点按 ranking 选择：
  - "rerank"（默认）：cross-encoder 重排候选池
  - "rrf"：按名次倒数融合两路召回，省掉重排开销
  - "none"：沿用召回顺序（bm25 在前、向量补位），用于对照

Router 支持两种模式（ROUTER_MODE）：
  - llm（默认）：with_structured_output 直接返回目标节点，异常时保守检索
  - always：一律检索，不做判断（兜底或对比用）

用法:
    uv run python -m taste_rag.rag.router
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from taste_rag.core import runtime
from taste_rag.chat.basic import DEFAULT_THREAD, DEFAULT_USER, thread_config
from taste_rag.rag.common import (
    RouteDecision,
    RouterState,
    make_answer_node,
    make_router_node,
    make_summarize_node,
)
from taste_rag.core.retrieval import (
    RRF_K,
    bm25_candidates,
    dedupe,
    format_context,
    rerank_candidates,
    rrf_fuse,
    vector_candidates,
)

DB_PATH = Path(__file__).resolve().parents[2] / ".router_chat.db"

BM25_K = 10  # 重排档：候选少，重排快
BM25_K_FUSION = 25  # 融合/无排序档：只看名次，多召回不亏
SEMANTIC_K = 15  # 参数扫描显示加深到 25 反而拉低 MRR
TOP_N = 8  # 参数扫描：top_n=5 只有 80% 召回，8 能到 90%
RANKINGS = ("rerank", "rrf", "none")


def build_graph(
    model,
    router_model,
    bm25,
    chunks,
    retriever,
    checkpointer=None,
    router_mode: str = "llm",
    ranking: str = "rerank",
    bm25_k: int | None = None,
    top_n: int = TOP_N,
    rrf_k: int = RRF_K,
    reranker=None,
):
    """构建并编译 chat 图。

    bm25_k 为 None 时按 ranking 取默认值：重排档 10，融合/无排序档 25。
    """
    if ranking not in RANKINGS:
        raise ValueError(f"ranking 必须是 {RANKINGS} 之一，收到 {ranking!r}")
    if ranking == "rerank" and reranker is None:
        raise ValueError("ranking='rerank' 需要传入 reranker")
    if bm25_k is None:
        bm25_k = BM25_K if ranking == "rerank" else BM25_K_FUSION

    structured_router = router_model.with_structured_output(RouteDecision)

    def bm25_retrieve(state: RouterState) -> dict:
        question = state["messages"][-1].text
        return {"bm25_docs": bm25_candidates(bm25, chunks, question, bm25_k)}

    def semantic_retrieve(state: RouterState) -> dict:
        question = state["messages"][-1].text
        return {"semantic_docs": vector_candidates(retriever, question)}

    def rank(state: RouterState) -> dict:
        """按 ranking 指定的方式挑出进入上下文的文档。"""
        question = state["messages"][-1].text
        groups = [state["bm25_docs"], state["semantic_docs"]]
        if ranking == "none":
            fused = dedupe(groups)[:top_n]
        elif ranking == "rerank":
            fused = rerank_candidates(reranker, dedupe(groups), question, top_n)
        else:
            fused = rrf_fuse(groups, top_n, rrf_k)
        return {"context": format_context(fused)}

    builder = StateGraph(RouterState)
    builder.add_node("summarize", make_summarize_node(model))
    builder.add_node("router", make_router_node(structured_router, router_mode))
    builder.add_node("bm25", bm25_retrieve)
    builder.add_node("semantic", semantic_retrieve)
    builder.add_node("rank", rank)
    builder.add_node("chatbot", make_answer_node(model))

    builder.add_edge(START, "summarize")
    builder.add_edge("summarize", "router")
    builder.add_conditional_edges(
        "router",
        lambda state: ["bm25", "semantic"] if state["search"] else "chatbot",
    )
    builder.add_edge("bm25", "rank")
    builder.add_edge("semantic", "rank")
    builder.add_edge("rank", "chatbot")
    builder.add_edge("chatbot", END)
    return builder.compile(checkpointer=checkpointer)


def main() -> None:
    runtime.silence_warnings()
    load_dotenv()

    stack = runtime.build_retrieval_stack(SEMANTIC_K, with_reranker=True)
    config = thread_config(DEFAULT_USER, DEFAULT_THREAD)

    with SqliteSaver.from_conn_string(str(DB_PATH)) as checkpointer:
        graph = build_graph(
            runtime.load_model(),
            runtime.load_router_model(),
            stack.bm25,
            stack.chunks,
            stack.retriever,
            checkpointer,
            os.environ.get("ROUTER_MODE", runtime.DEFAULT_ROUTER_MODE),
            reranker=stack.reranker,
        )
        runtime.run_cli(lambda question: runtime.stream_answer(graph, config, question))


if __name__ == "__main__":
    main()
