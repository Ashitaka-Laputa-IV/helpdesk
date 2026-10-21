"""三种"检索后排序"方案对比：无排序 / rerank / RRF。

同一召回池（bm25_k + sem_k，按 (来源, 文本) 去重）下比较：

  - 无排序：直接沿用召回顺序（bm25 在前、向量补位），不做事后排序
  - rerank：cross-encoder 对候选池重新打分
  - RRF   ：按名次倒数融合两路结果（rrf_k）

指标：Recall@topN、MRR（正解首次出现名次的倒数均值，文档级去重后计算）
耗时：rerank 实测；无排序为 0；RRF 只统计融合本身开销

用法:
    uv run python tests/compare_ranking.py
    uv run python tests/compare_ranking.py --repeat 2
"""

import argparse
import json
import os
import statistics
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_reranker.rerankers import init_reranker

from taste_rag.core.bm25 import build_index, search
from taste_rag.core.corpus import load_chunks
from taste_rag.core.vector_store import COLLECTION, INDEX_DIR
from taste_rag.rag.workflow import BM25_K_FUSION, RRF_K, SEMANTIC_K

EVAL_PATH = Path(__file__).resolve().parents[1] / "data" / "eval" / "queries.json"
TOP_NS = (5, 8, 10)
TIERS = ("title", "paraphrase")


def top_docs(docs: list[dict], top_n: int) -> list[dict]:
    """按文档去重后取前 top_n，避免同一文档的多个分块占位。"""
    seen, picked = set(), []
    for doc in docs:
        if doc["source"] in seen:
            continue
        seen.add(doc["source"])
        picked.append(doc)
        if len(picked) == top_n:
            break
    return picked


def rrf_order(bm25_docs: list[dict], semantic_docs: list[dict], rrf_k: int) -> list[dict]:
    """按名次倒数融合，返回文档顺序（保留每篇文档的首次出现）。"""
    scores: dict[str, float] = {}
    first: dict[str, dict] = {}
    for ranked in (bm25_docs, semantic_docs):
        for rank, doc in enumerate(ranked, 1):
            source = doc["source"]
            scores[source] = scores.get(source, 0.0) + 1.0 / (rrf_k + rank)
            first.setdefault(source, doc)
    return [first[source] for source, _ in sorted(scores.items(), key=lambda x: -x[1])]


def measure(rows: list[dict], orders: list[list[dict]], top_n: int) -> tuple[float, float]:
    """按给定顺序统计 Recall@topN 与 MRR。"""
    hits, rr = 0, 0.0
    for row, ordered in zip(rows, orders):
        picked = top_docs(ordered, top_n)
        for rank, doc in enumerate(picked, 1):
            if doc["source"] == row["target"]:
                hits += 1
                rr += 1.0 / rank
                break
    return hits / len(rows) * 100, rr / len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="检索后排序方案对比")
    parser.add_argument("--repeat", type=int, default=1, help="重复次数（用于耗时平均）")
    args = parser.parse_args()

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

    queries = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    print(f"召回池: bm25_k={BM25_K_FUSION} + sem_k={SEMANTIC_K} | RRF k={RRF_K}")

    for tier in TIERS:
        items = [q for q in queries if q["tier"] == tier]

        # 召回：每条查询取一次 bm25 与向量结果
        rows = []
        for item in items:
            hits = search(bm25, chunks, item["query"], BM25_K_FUSION)
            docs = retriever.invoke(item["query"])
            rows.append(
                {
                    "query": item["query"],
                    "target": item["target"],
                    "bm25": [{"source": s, "text": t} for _, s, t in hits],
                    "vector": [
                        {"source": d.metadata["source"], "text": d.page_content}
                        for d in docs
                    ],
                }
            )

        # 无排序：bm25 在前、向量补位，去重后即顺序
        base_orders = []
        for row in rows:
            merged, seen = [], set()
            for doc in row["bm25"] + row["vector"]:
                key = (doc["source"], doc["text"])
                if key not in seen:
                    seen.add(key)
                    merged.append(doc)
            base_orders.append(merged)

        # RRF：融合两路名次，记录融合本身耗时
        rrf_orders, rrf_ms = [], []
        for row in rows:
            started = time.perf_counter()
            rrf_orders.append(rrf_order(row["bm25"], row["vector"], RRF_K))
            rrf_ms.append((time.perf_counter() - started) * 1000)

        # rerank：对候选池重排，重复 args.repeat 次取中位耗时
        rerank_orders, rerank_ms = [], []
        for row in rows:
            pool, seen = [], set()
            for doc in row["bm25"] + row["vector"]:
                key = (doc["source"], doc["text"])
                if key not in seen:
                    seen.add(key)
                    pool.append(doc)
            durations = []
            for _ in range(args.repeat):
                started = time.perf_counter()
                ranked = reranker.compress_documents(
                    [
                        Document(page_content=d["text"], metadata={"source": d["source"]})
                        for d in pool
                    ],
                    query=row["query"],
                )
                durations.append((time.perf_counter() - started) * 1000)
                order = [{"source": d.metadata["source"]} for d in ranked]
            rerank_ms.append(statistics.median(durations))
            rerank_orders.append(order)

        print(f"\n===== {tier}（{len(items)} 条）=====")
        header = (
            f"{'topN':>5}"
            f"{'无排序 Recall':>14}{'MRR':>8}"
            f"{'rerank Recall':>15}{'MRR':>8}"
            f"{'RRF Recall':>12}{'MRR':>8}"
        )
        print(header)
        for top_n in TOP_NS:
            base = measure(rows, base_orders, top_n)
            rrk = measure(rows, rerank_orders, top_n)
            fusion = measure(rows, rrf_orders, top_n)
            print(
                f"{top_n:>5}"
                f"{base[0]:>13.1f}%{base[1]:>8.3f}"
                f"{rrk[0]:>14.1f}%{rrk[1]:>8.3f}"
                f"{fusion[0]:>11.1f}%{fusion[1]:>8.3f}"
            )
        print(
            f"  耗时: 无排序 0ms | rerank 中位 {statistics.median(rerank_ms):.0f}ms"
            f" | RRF 融合中位 {statistics.median(rrf_ms):.2f}ms"
        )


if __name__ == "__main__":
    main()
