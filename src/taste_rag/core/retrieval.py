"""检索原语：双路召回、去重、融合、重排、上下文拼接。

各 rag_* 实现只在「什么时候检索」「怎么排序」上不同，
召回与格式化的细节都收敛到这里。
"""

from langchain_core.documents import Document

from taste_rag.core.bm25 import search

RRF_K = 20  # 平滑常数；参数扫描中 20 优于常用的 60


def bm25_candidates(bm25, chunks, query: str, k: int) -> list[dict]:
    """bm25s 召回，统一成 {"source", "text"}。"""
    return [{"source": s, "text": t} for _, s, t in search(bm25, chunks, query, k)]


def sem_candidates(retriever, query: str) -> list[dict]:
    """语义（向量）召回，统一成 {"source", "text"}。"""
    return [
        {"source": doc.metadata["source"], "text": doc.page_content}
        for doc in retriever.invoke(query)
    ]


def dedupe(groups) -> list[dict]:
    """按 (来源, 原文) 去重，保持各组原有顺序。"""
    merged: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for group in groups:
        for doc in group:
            key = (doc["source"], doc["text"])
            if key not in seen:
                seen.add(key)
                merged.append(doc)
    return merged


def rerank(reranker, candidates, query: str, top_n: int) -> list[dict]:
    """用 cross-encoder 按原始问题重新打分，取前 top_n。"""
    if not candidates:
        return []
    documents = [
        Document(page_content=c["text"], metadata={"source": c["source"]})
        for c in candidates
    ]
    return [
        {"source": doc.metadata["source"], "text": doc.page_content}
        for doc in reranker.compress_documents(documents, query=query)[:top_n]
    ]


def rrf_fuse(ranked_lists, top_n: int, rrf_k: int = RRF_K) -> list[dict]:
    """按名次倒数融合多路结果，返回 top_n。

    只依赖排名、不依赖分数量纲，因此 bm25 的概率分与向量的余弦分可以直接混用。
    rrf_k 越大，靠后名次的衰减越平缓；分数相同时保持首次出现的先后，因为 sorted 稳定。
    """
    scores: dict[tuple[str, str], float] = {}
    docs: dict[tuple[str, str], dict] = {}
    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked, 1):
            key = (doc["source"], doc["text"])
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)
            docs[key] = doc
    ordered = sorted(scores.items(), key=lambda item: -item[1])[:top_n]
    return [docs[key] for key, _ in ordered]


def format_context(candidates) -> str:
    """拼成提示词里的资料块：每段带来源文件名。"""
    return "\n\n".join(f"[{c['source']}] {c['text']}" for c in candidates)
