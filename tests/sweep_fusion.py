"""参数扫描：bm25_k / semantic_k / rrf_k / top_n 对召回质量与耗时的影响。

设计要点：
  - 每条查询只做一次召回（bm25 取 30、向量取 30），后续各种截断/融合配置离线计算；
  - rerank 基线按「候选池」各算一次（rerank 排名与 top_n 无关），再按 top_n 切片；
  - 指标：Recall@top_n、MRR（正解首次出现名次的倒数均值）。

用法:
    uv run python tests/sweep_fusion.py
    uv run python tests/sweep_fusion.py --tier title
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

from taste_rag.bm25 import build_index, search
from taste_rag.corpus import load_chunks
from taste_rag.vector_store import COLLECTION, INDEX_DIR

POOL_K = 30  # 一次召回的深度，需覆盖所有配置里最大的 k
EVAL_PATH = Path(__file__).resolve().parents[1] / "data" / "eval" / "queries.json"

BM25_KS = (10, 20, 25)
SEMANTIC_KS = (15, 25)
RRF_KS = (20, 60)
TOP_NS = (5, 8, 10)


def rrf_order(bm25_docs, semantic_docs, rrf_k):
    """按名次倒数融合，返回去重后的文档顺序（每个文档只保留首次出现）。"""
    scores: dict[str, float] = {}
    first: dict[str, dict] = {}
    for ranked in (bm25_docs, semantic_docs):
        for rank, doc in enumerate(ranked, 1):
            source = doc["source"]
            scores[source] = scores.get(source, 0.0) + 1.0 / (rrf_k + rank)
            first.setdefault(source, doc)
    return [first[source] for source, _ in sorted(scores.items(), key=lambda x: -x[1])]


def dedupe_by_source(docs):
    """按文档去重，保留每篇文档的首次出现。"""
    seen, ordered = set(), []
    for doc in docs:
        if doc["source"] not in seen:
            seen.add(doc["source"])
            ordered.append(doc)
    return ordered


def evaluate(rows, picker, top_n):
    """用 picker(样本) 给出排好序的文档列表，统计 recall@top_n 与 MRR。"""
    hits, rr = 0, 0.0
    for row in rows:
        ordered = picker(row)[:top_n]
        for rank, doc in enumerate(ordered, 1):
            if doc["source"] == row["target"]:
                hits += 1
                rr += 1.0 / rank
                break
    return hits / len(rows) * 100, rr / len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="融合参数扫描")
    parser.add_argument("--tier", default="paraphrase", choices=["title", "paraphrase"])
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
    ).as_retriever(search_kwargs={"k": POOL_K})
    reranker = init_reranker(
        os.environ["RERANKER_MODEL"], base_url=os.environ["RERANKER_BASE_URL"]
    )

    queries = [r for r in json.loads(EVAL_PATH.read_text(encoding="utf-8")) if r["tier"] == args.tier]
    print(f"评测集 [{args.tier}] {len(queries)} 条")

    # 一次召回，缓存 30 条
    rows = []
    bm25_ms, vec_ms = [], []
    for item in queries:
        started = time.perf_counter()
        hits = search(bm25, chunks, item["query"], POOL_K)
        bm25_ms.append((time.perf_counter() - started) * 1000)
        started = time.perf_counter()
        docs = retriever.invoke(item["query"])
        vec_ms.append((time.perf_counter() - started) * 1000)
        rows.append(
            {
                "query": item["query"],
                "target": item["target"],
                "bm25": [{"source": s, "text": t} for _, s, t in hits],
                "vector": [
                    {"source": d.metadata["source"], "text": d.page_content} for d in docs
                ],
            }
        )
    print(
        f"召回耗时: bm25 中位 {statistics.median(bm25_ms):.1f}ms | "
        f"向量 中位 {statistics.median(vec_ms):.1f}ms"
    )

    # 1) RRF 参数扫描
    print("\n===== RRF 融合扫描 =====")
    print(f"{'bm25_k':>7}{'sem_k':>7}{'rrf_k':>7}{'top_n':>7}{'Recall':>9}{'MRR':>8}")
    best = []
    for bm_k in BM25_KS:
        for sem_k in SEMANTIC_KS:
            for rrf_k in RRF_KS:
                picker = lambda row, b=bm_k, s=sem_k, r=rrf_k: rrf_order(
                    row["bm25"][:b], row["vector"][:s], r
                )
                for top_n in TOP_NS:
                    recall, mrr = evaluate(rows, picker, top_n)
                    best.append((mrr, recall, bm_k, sem_k, rrf_k, top_n))
                    print(
                        f"{bm_k:>7}{sem_k:>7}{rrf_k:>7}{top_n:>7}"
                        f"{recall:>8.1f}%{mrr:>8.3f}"
                    )

    # 2) rerank 基线：每个候选池只重排一次，再按 top_n 切片
    print("\n===== rerank 基线（按文档去重后截断）=====")
    print(f"{'bm25_k':>7}{'sem_k':>7}{'top_n':>7}{'Recall':>9}{'MRR':>8}{'rerank':>9}")
    rerank_rows = []
    for bm_k in BM25_KS:
        for sem_k in SEMANTIC_KS:
            ordered_cache = []
            rerank_ms = []
            for row in rows:
                pool, seen = [], set()
                for doc in row["bm25"][:bm_k] + row["vector"][:sem_k]:
                    key = (doc["source"], doc["text"])
                    if key not in seen:
                        seen.add(key)
                        pool.append(doc)
                started = time.perf_counter()
                ranked = reranker.compress_documents(
                    [
                        Document(page_content=d["text"], metadata={"source": d["source"]})
                        for d in pool
                    ],
                    query=row["query"],
                )
                rerank_ms.append((time.perf_counter() - started) * 1000)
                ordered_cache.append(
                    [{"source": d.metadata["source"]} for d in ranked]
                )
            for top_n in TOP_NS:
                hits, rr = 0, 0.0
                for row, ordered in zip(rows, ordered_cache):
                    subset = dedupe_by_source(ordered)[:top_n]
                    for rank, doc in enumerate(subset, 1):
                        if doc["source"] == row["target"]:
                            hits += 1
                            rr += 1.0 / rank
                            break
                recall = hits / len(rows) * 100
                mrr = rr / len(rows)
                rerank_rows.append((mrr, recall, bm_k, sem_k, top_n))
                print(
                    f"{bm_k:>7}{sem_k:>7}{top_n:>7}{recall:>8.1f}%{mrr:>8.3f}"
                    f"{statistics.median(rerank_ms):>8.0f}ms"
                )

    print("\n===== 最优配置 =====")
    for label, pool_rows in (("RRF", best), ("rerank", rerank_rows)):
        top = sorted(pool_rows, reverse=True)[:3]
        for mrr, recall, bm_k, sem_k, *rest in top:
            extra = f"rrf_k={rest[0]}" if label == "RRF" else ""
            top_n = rest[-1]
            print(
                f"  {label:6s} bm25_k={bm_k} sem_k={sem_k} {extra} top_n={top_n}"
                f" -> Recall {recall:.1f}% MRR {mrr:.3f}"
            )


if __name__ == "__main__":
    main()
