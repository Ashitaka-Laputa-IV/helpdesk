"""检索链路基准：bm25 / 向量 / rerank 的耗时与召回率。

评测集自动构建：从语料里抽取标题唯一的文档，用其 H1 标题当查询，正解就是该文档本身。
每个查询重复跑 R 次取平均，分别统计各阶段的耗时与 Recall@K。

指标口径：
  - Recall@K：正解文档是否出现在该阶段的 top-K 里（按文件路径判断）
  - 候选池召回：bm25 ∪ 向量的去重结果里是否含正解（rerank 的上限）
  - 条件召回：rerank 命中数 / 候选池命中数（衡量重排本身有没有把正解挤出去）

用法:
    uv run python tests/benchmark_retrieval.py
    uv run python tests/benchmark_retrieval.py --queries 40 --repeat 3
"""

import argparse
import os
import random
import re
import statistics
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_reranker.rerankers import init_reranker

from helpdesk.bm25 import build_index, load_chunks, search
from helpdesk.rag import COLLECTION, DOCS_DIR, INDEX_DIR
from helpdesk.rag_agent import BM25_K, TOP_K
from helpdesk.router_chat import SEMANTIC_K

SEED = 42


def build_queries(limit: int) -> list[dict]:
    """抽样：标题唯一的文档，标题当查询、文件路径当正解。"""
    titles: dict[str, list[str]] = {}
    for path in sorted(DOCS_DIR.rglob("*.md")):
        match = re.search(r"^#\s+(.+)$", path.read_text(encoding="utf-8"), re.M)
        if match:
            titles.setdefault(match.group(1).strip(), []).append(
                str(path.relative_to(DOCS_DIR))
            )

    pool = [
        {"query": title, "target": sources[0]}
        for title, sources in titles.items()
        if len(sources) == 1 and len(title.split()) >= 2
    ]
    random.Random(SEED).shuffle(pool)
    return pool[:limit]


def percentile(values: list[float], ratio: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(len(ordered) * ratio))
    return ordered[index]


def main() -> None:
    parser = argparse.ArgumentParser(description="检索链路基准")
    parser.add_argument("--queries", type=int, default=40, help="查询条数")
    parser.add_argument("--repeat", type=int, default=3, help="每条查询重复次数")
    parser.add_argument("--bm25-k", type=int, default=BM25_K, help="bm25 召回条数")
    parser.add_argument("--semantic-k", type=int, default=SEMANTIC_K, help="向量召回条数")
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
    ).as_retriever(search_kwargs={"k": args.semantic_k})
    reranker = init_reranker(
        os.environ["RERANKER_MODEL"], base_url=os.environ["RERANKER_BASE_URL"]
    )

    queries = build_queries(args.queries)
    # 预热：避免首次调用的建连/加载计入统计
    warm = queries[0]["query"]
    search(bm25, chunks, warm, args.bm25_k)
    retriever.invoke(warm)
    reranker.compress_documents(
        [Document(page_content=t, metadata={"source": s}) for s, t in chunks[:25]],
        query=warm,
    )

    stats = {
        name: {"time": [], "hit": 0}
        for name in ("bm25", "vector", "pool", "base", "rerank")
    }
    total_time: list[float] = []

    for _ in range(args.repeat):
        for item in queries:
            query, target = item["query"], item["target"]

            started = time.perf_counter()
            hits = search(bm25, chunks, query, args.bm25_k)
            stats["bm25"]["time"].append((time.perf_counter() - started) * 1000)
            bm25_sources = [source for _, source, _ in hits]

            started = time.perf_counter()
            vector_docs = retriever.invoke(query)
            stats["vector"]["time"].append((time.perf_counter() - started) * 1000)
            vector_sources = [doc.metadata["source"] for doc in vector_docs]

            # 去重合并（与 rag_agent / router_chat 的候选池口径一致）
            candidates, seen = [], set()
            for _, source, text in hits:
                key = (source, text)
                if key not in seen:
                    seen.add(key)
                    candidates.append({"source": source, "text": text})
            for doc in vector_docs:
                key = (doc.metadata["source"], doc.page_content)
                if key not in seen:
                    seen.add(key)
                    candidates.append(
                        {"source": key[0], "text": key[1]}
                    )

            started = time.perf_counter()
            ranked = reranker.compress_documents(
                [Document(page_content=c["text"], metadata={"source": c["source"]}) for c in candidates],
                query=query,
            )[:TOP_K]
            stats["rerank"]["time"].append((time.perf_counter() - started) * 1000)
            total_time.append(
                stats["bm25"]["time"][-1]
                + stats["vector"]["time"][-1]
                + stats["rerank"]["time"][-1]
            )

            stats["bm25"]["hit"] += target in bm25_sources
            stats["vector"]["hit"] += target in vector_sources
            stats["pool"]["hit"] += target in {c["source"] for c in candidates}
            # 不重排基线：直接取候选池前 TOP_K 条（bm25 在前，向量补位）
            stats["base"]["hit"] += target in {c["source"] for c in candidates[:TOP_K]}
            stats["rerank"]["hit"] += target in {d.metadata["source"] for d in ranked}

    samples = len(queries) * args.repeat
    print(f"查询 {len(queries)} 条 × 重复 {args.repeat} 次 = {samples} 次采样")
    print(
        f"语料 {len(chunks)} 块 | bm25 K={args.bm25_k} | "
        f"向量 K={args.semantic_k} | rerank 取 {TOP_K}"
    )
    print()
    print(f"{'阶段':<14}{'平均':>9}{'中位':>9}{'P95':>9}{'召回率':>10}")
    labels = {
        "bm25": f"bm25 top{args.bm25_k}",
        "vector": f"向量 top{args.semantic_k}",
        "pool": "候选池(去重)",
        "base": f"不重排 top{TOP_K}",
        "rerank": f"rerank top{TOP_K}",
    }
    for name in ("bm25", "vector", "pool", "base", "rerank"):
        times = stats[name]["time"]
        time_cell = f"{statistics.mean(times):.1f}ms" if times else "-"
        median_cell = f"{statistics.median(times):.1f}ms" if times else "-"
        p95_cell = f"{percentile(times, 0.95):.1f}ms" if times else "-"
        recall = stats[name]["hit"] / samples * 100
        print(
            f"{labels[name]:<14}{time_cell:>9}{median_cell:>9}{p95_cell:>9}{recall:>9.1f}%"
        )

    print()
    print(f"整条链路（bm25 + 向量 + rerank 串行）: 平均 {statistics.mean(total_time):.0f}ms | 中位 {statistics.median(total_time):.0f}ms")
    pool_recall = stats["pool"]["hit"]
    if pool_recall:
        print(
            f"重排保真度: rerank 命中 {stats['rerank']['hit']} / 候选池命中 {pool_recall}"
            f" = {stats['rerank']['hit'] / pool_recall * 100:.1f}%"
        )


if __name__ == "__main__":
    main()
