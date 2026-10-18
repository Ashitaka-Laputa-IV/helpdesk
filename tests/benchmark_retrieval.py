"""检索链路基准：bm25 / 向量 / rerank 的耗时与召回质量。

评测集两种来源：
  1) 内置标题集：文档 H1 标题当查询（简单档，字面重合多）。
  2) `--eval-set` 指定的 JSON（如 data/eval/queries.json），含 title / paraphrase 两档。

指标：
  - Recall@K：正解文档是否出现在该阶段 top-K
  - MRR：正解首次出现排名的倒数均值（1/rank），衡量排序质量而不只看"有没有进前 K"
  - 候选池召回：bm25 ∪ 向量的并集是否含正解，即 rerank 的理论上限
  - 不重排基线：直接取候选池前 5 条（bm25 在前、向量补位）
  - 重排保真度：rerank 命中 / 候选池命中

用法:
    uv run python tests/benchmark_retrieval.py
    uv run python tests/benchmark_retrieval.py --eval-set data/eval/queries.json --tier paraphrase
    uv run python tests/benchmark_retrieval.py --bm25-k 10 --semantic-k 15 --repeat 3
"""

import argparse
import json
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


def build_title_queries(limit: int) -> list[dict]:
    """内置简单档：标题唯一的文档，标题当查询、文件路径当正解。"""
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


def load_eval_set(path: str, tier: str | None) -> list[dict]:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    if tier:
        rows = [row for row in rows if row["tier"] == tier]
    return [{"query": row["query"], "target": row["target"]} for row in rows]


def percentile(values: list[float], ratio: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * ratio))]


def rank_of(sources: list[str], target: str) -> int:
    """正解在结果里的名次（1 起），没出现返回 0。"""
    for index, source in enumerate(sources, 1):
        if source == target:
            return index
    return 0


def top_distinct(sources: list[str], k: int) -> list[str]:
    """按文档去重后取前 k：避免同一文档的多个分块挤占名额。"""
    picked: list[str] = []
    seen: set[str] = set()
    for source in sources:
        if source not in seen:
            seen.add(source)
            picked.append(source)
        if len(picked) == k:
            break
    return picked


def rrf_fuse(bm25_sources: list[str], vector_sources: list[str], top: int) -> list[str]:
    """Reciprocal Rank Fusion：只按名次融合，无需调分数尺度。

    作为"不花 rerank 那 500ms"的公平基线——直接取候选池前 5 会偏向 bm25，太悲观。
    """
    scores: dict[str, float] = {}
    for sources in (bm25_sources, vector_sources):
        for rank, source in enumerate(sources, 1):
            scores[source] = scores.get(source, 0.0) + 1.0 / (60 + rank)
    return [source for source, _ in sorted(scores.items(), key=lambda x: -x[1])[:top]]


def main() -> None:
    parser = argparse.ArgumentParser(description="检索链路基准")
    parser.add_argument("--queries", type=int, default=40, help="内置标题集的条数")
    parser.add_argument("--repeat", type=int, default=3, help="每条查询重复次数")
    parser.add_argument("--bm25-k", type=int, default=BM25_K, help="bm25 召回条数")
    parser.add_argument("--semantic-k", type=int, default=SEMANTIC_K, help="向量召回条数")
    parser.add_argument("--eval-set", help="评测集 JSON 路径，不传则用内置标题集")
    parser.add_argument("--tier", choices=["title", "paraphrase"], help="只跑某一档")
    parser.add_argument("--top-k", type=int, default=TOP_K, help="rerank 后进入上下文的条数")
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

    if args.eval_set:
        queries = load_eval_set(args.eval_set, args.tier)
        source_desc = f"{Path(args.eval_set).name}" + (f" [{args.tier}]" if args.tier else "")
    else:
        queries = build_title_queries(args.queries)
        source_desc = "内置标题集"

    # 预热，避免首次建连/加载计入统计
    warm = queries[0]["query"]
    search(bm25, chunks, warm, args.bm25_k)
    retriever.invoke(warm)
    reranker.compress_documents(
        [Document(page_content=t, metadata={"source": s}) for s, t in chunks[:25]],
        query=warm,
    )

    stages = ("bm25", "vector", "pool", "base", "rrf", "rerank", "rerank_uniq")
    stats = {name: {"time": [], "hit": 0, "rr": 0.0} for name in stages}
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
                    candidates.append({"source": key[0], "text": key[1]})

            started = time.perf_counter()
            # 保留完整排名，用于"按文档去重"的对照
            ordered = reranker.compress_documents(
                [
                    Document(page_content=c["text"], metadata={"source": c["source"]})
                    for c in candidates
                ],
                query=query,
            )
            ranked = ordered[: args.top_k]
            stats["rerank"]["time"].append((time.perf_counter() - started) * 1000)
            total_time.append(
                stats["bm25"]["time"][-1]
                + stats["vector"]["time"][-1]
                + stats["rerank"]["time"][-1]
            )

            results = {
                "bm25": bm25_sources,
                "vector": vector_sources,
                "pool": [c["source"] for c in candidates],
                "base": [c["source"] for c in candidates[: args.top_k]],
                "rrf": rrf_fuse(bm25_sources, vector_sources, args.top_k),
                "rerank": [doc.metadata["source"] for doc in ranked],
                "rerank_uniq": top_distinct(
                    [doc.metadata["source"] for doc in ordered], args.top_k
                ),
            }
            for name, sources in results.items():
                rank = rank_of(sources, target)
                if rank:
                    stats[name]["hit"] += 1
                    stats[name]["rr"] += 1.0 / rank

    samples = len(queries) * args.repeat
    print(f"评测集 {source_desc} | 查询 {len(queries)} 条 × 重复 {args.repeat} 次 = {samples} 次采样")
    print(
        f"语料 {len(chunks)} 块 | bm25 K={args.bm25_k} | "
        f"向量 K={args.semantic_k} | rerank 取 {args.top_k}"
    )
    print()
    print(f"{'阶段':<13}{'平均':>9}{'中位':>9}{'P95':>9}{'Recall':>9}{'MRR':>8}")

    labels = {
        "bm25": f"bm25 top{args.bm25_k}",
        "vector": f"向量 top{args.semantic_k}",
        "pool": "候选池",
        "base": f"不重排 top{args.top_k}",
        "rrf": f"RRF top{args.top_k}",
        "rerank": f"rerank top{args.top_k}",
        "rerank_uniq": f"rerank 去重top{args.top_k}",
    }
    for name in stages:
        times = stats[name]["time"]
        time_cell = f"{statistics.mean(times):.1f}ms" if times else "-"
        median_cell = f"{statistics.median(times):.1f}ms" if times else "-"
        p95_cell = f"{percentile(times, 0.95):.1f}ms" if times else "-"
        recall = stats[name]["hit"] / samples * 100
        mrr = stats[name]["rr"] / samples
        print(
            f"{labels[name]:<13}{time_cell:>9}{median_cell:>9}{p95_cell:>9}"
            f"{recall:>8.1f}%{mrr:>8.3f}"
        )

    print()
    print(
        f"整条链路（bm25 + 向量 + rerank 串行）: "
        f"平均 {statistics.mean(total_time):.0f}ms | 中位 {statistics.median(total_time):.0f}ms"
    )
    pool_hits = stats["pool"]["hit"]
    if pool_hits:
        print(
            f"重排保真度: rerank 命中 {stats['rerank']['hit']} / 候选池命中 {pool_hits}"
            f" = {stats['rerank']['hit'] / pool_hits * 100:.1f}%"
        )


if __name__ == "__main__":
    main()
