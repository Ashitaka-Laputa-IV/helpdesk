"""bm25s 稀疏检索：建索引与查询。

语料读取与切块在 corpus.py，本模块只负责索引本身。

用法:
    uv run python -m taste_rag.core.bm25 "how to define a tool"
    uv run python -m taste_rag.core.bm25            # 交互模式
"""

import argparse

import bm25s

from taste_rag.core.corpus import DOCS_DIR, load_chunks

TOP_K = 5
PREVIEW_CHARS = 20  # 结果预览只显示前多少个字符


def build_index(chunks: list[tuple[str, str]]):
    """对语料分词并建立 BM25 索引。"""
    corpus_tokens = bm25s.tokenize([text for _, text in chunks], show_progress=False)
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens, show_progress=False)
    return retriever


def search(retriever, chunks, query: str, k: int = TOP_K):
    """返回 [(得分, 来源, 文本)]，已按得分降序。"""
    query_tokens = bm25s.tokenize(query, show_progress=False)
    indices, scores = retriever.retrieve(query_tokens, k=k, show_progress=False)
    return [
        (float(scores[0][i]), chunks[j][0], chunks[j][1])
        for i, j in enumerate(indices[0])
    ]


def show(retriever, chunks, query: str, k: int = TOP_K) -> None:
    print(f"\n查询: {query}")
    for rank, (score, source, text) in enumerate(search(retriever, chunks, query, k), 1):
        preview = " ".join(text.split())
        suffix = "..." if len(preview) > PREVIEW_CHARS else ""
        print(f"  {rank}. {score:6.3f}  {source}")
        print(f"     {preview[:PREVIEW_CHARS]}{suffix}")


def main() -> None:
    parser = argparse.ArgumentParser(description="BM25 关键词检索 demo")
    parser.add_argument("query", nargs="?", help="查询词；不传则进入交互模式")
    parser.add_argument("-k", type=int, default=TOP_K, help="返回条数")
    args = parser.parse_args()

    chunks = load_chunks()
    retriever = build_index(chunks)
    print(f"已索引 {len(chunks)} 个分块（来自 {DOCS_DIR}）")

    if args.query:
        show(retriever, chunks, args.query, args.k)
        return

    print("输入查询词，exit / quit 结束。")
    while True:
        try:
            query = input("Q: ")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in {"exit", "quit"}:
            break
        if query.strip():
            show(retriever, chunks, query, args.k)


if __name__ == "__main__":
    main()
