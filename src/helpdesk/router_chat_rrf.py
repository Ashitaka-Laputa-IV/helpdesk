"""并行召回 + RRF 融合的问答 chat（不使用 reranker）。

图结构：START -> summarize -> router
         router --检索--> bm25(10) ─┐
         router --检索--> semantic(15) ─> rrf(融合取 topN) -> chatbot
         router --不检索--------------------> chatbot
         chatbot -> END

与 router_chat.py 的唯一区别：把 combine + rerank 两个节点换成 rrf 一个节点，
用名次倒数融合（Reciprocal Rank Fusion）挑 topN，省掉 rerank 的数百毫秒开销。

用法:
    uv run python src/helpdesk/router_chat_rrf.py
"""

import os
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, SystemMessage
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from helpdesk import DEFAULT_THREAD, DEFAULT_USER, silence_beta_warning, thread_config
from helpdesk.bm25 import build_index, load_chunks, search
from helpdesk.rag import COLLECTION, INDEX_DIR
from helpdesk.router_chat import (
    PLAIN_PROMPT,
    SYSTEM_PROMPT,
    ChatState,
    RouteDecision,
    needs_retrieval,
)
from helpdesk.summarize import compress_history

DB_PATH = Path(__file__).resolve().parents[2] / ".router_chat_rrf.db"
# 以下取值来自 tests/sweep_fusion.py 的参数扫描（英文改写档，30 条）
BM25_K = 25  # 比重排版更深：RRF 只看名次，多召回不亏
SEMANTIC_K = 15  # 扫描显示加深到 25 反而拉低 MRR
RRF_K = 20  # 平滑常数，扫描中 20 优于常用的 60
TOP_N = 8  # 融合后进入上下文的条数


def rrf_rank(bm25_docs: list[dict], semantic_docs: list[dict], top_n: int) -> list[dict]:
    """按名次倒数融合两路召回，返回 top_n。

    只依赖排名、不依赖分数量纲，因此 bm25 的概率分与向量的余弦分可以直接混用。
    """
    scores: dict[tuple[str, str], float] = {}
    docs: dict[tuple[str, str], dict] = {}
    for ranked in (bm25_docs, semantic_docs):
        for rank, doc in enumerate(ranked, 1):
            key = (doc["source"], doc["text"])
            scores[key] = scores.get(key, 0.0) + 1.0 / (RRF_K + rank)
            docs[key] = doc
    ordered = sorted(scores.items(), key=lambda item: -item[1])[:top_n]
    return [docs[key] for key, _ in ordered]


def build_graph(
    model,
    router_model,
    bm25,
    chunks,
    retriever,
    checkpointer=None,
    router_mode: str = "llm",
    top_n: int = TOP_N,
    ranking: str = "rrf",
    reranker=None,
):
    """构建并编译 chat 图。

    ranking 决定检索后如何排序，用于横向评测：
      - "rrf"（默认）：按名次倒数融合
      - "none"：沿用召回顺序（bm25 在前、向量补位）
      - "rerank"：用传入的 reranker 重新打分

    router_mode 决定要不要检索：
      - "llm"（默认）：交给模型判断，寒暄与业务问题一视同仁
      - "always"：一律检索（用于对比"无判断"方案）
    """
    structured_router = router_model.with_structured_output(RouteDecision)

    def summarize(state: ChatState) -> dict:
        """历史超预算时把较早的对话压成摘要，控制上下文长度。"""
        updates = compress_history(model, state["messages"])
        return {"messages": updates} if updates else {}

    def router(state: ChatState) -> dict:
        started = time.perf_counter()
        # 不区分问题类型：llm 模式全部交给模型判断，always 模式一律检索
        if router_mode == "llm":
            search_needed = needs_retrieval(structured_router, state["messages"])
        else:
            search_needed = True
        cost = time.perf_counter() - started
        print(
            f"[router] {'检索' if search_needed else '不检索'}"
            f"  {cost * 1000:.0f}ms  ({router_mode})"
        )
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

    def rank(state: ChatState) -> dict:
        """按 ranking 指定的方式挑出进入上下文的文档。"""
        question = state["messages"][-1].text
        if ranking == "none":
            merged, seen = [], set()
            for doc in state["bm25_docs"] + state["semantic_docs"]:
                key = (doc["source"], doc["text"])
                if key not in seen:
                    seen.add(key)
                    merged.append(doc)
            fused = merged[:top_n]
        elif ranking == "rerank":
            documents = [
                Document(page_content=d["text"], metadata={"source": d["source"]})
                for d in state["bm25_docs"] + state["semantic_docs"]
            ]
            fused = [
                {"source": doc.metadata["source"], "text": doc.page_content}
                for doc in reranker.compress_documents(documents, query=question)[:top_n]
            ]
        else:
            fused = rrf_rank(state["bm25_docs"], state["semantic_docs"], top_n)
        return {
            "context": "\n\n".join(f"[{d['source']}] {d['text']}" for d in fused)
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
    builder.add_node("summarize", summarize)
    builder.add_node("router", router)
    builder.add_node("bm25", bm25_retrieve)
    builder.add_node("semantic", semantic_retrieve)
    builder.add_node("rank", rank)
    builder.add_node("chatbot", chatbot)

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
    print(f"已索引 {len(chunks)} 个分块（RRF 融合，不重排）")

    model = init_chat_model(os.environ["MODEL"])
    router_model = init_chat_model(
        os.environ.get("ROUTER_MODEL", os.environ["MODEL"]),
        base_url=os.environ.get("ROUTER_BASE_URL", os.environ["OPENAI_API_BASE"]),
        api_key=os.environ["OPENAI_API_KEY"],
        temperature=0,
        max_tokens=128,
    )
    config = thread_config(DEFAULT_USER, DEFAULT_THREAD)

    with SqliteSaver.from_conn_string(str(DB_PATH)) as checkpointer:
        graph = build_graph(
            model,
            router_model,
            bm25,
            chunks,
            retriever,
            checkpointer,
            os.environ.get("ROUTER_MODE", "llm"),
        )

        print("输入问题开始问答，输入 exit / quit 结束。")
        while True:
            try:
                question = input("Q: ")
            except (EOFError, KeyboardInterrupt):
                break
            if question.strip().lower() in {"exit", "quit"}:
                break

            result = graph.invoke(
                {"messages": [HumanMessage(content=question)]}, config
            )
            print(f"A: {result['messages'][-1].text}")


if __name__ == "__main__":
    main()
