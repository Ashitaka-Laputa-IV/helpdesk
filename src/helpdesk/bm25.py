"""BM25 关键词搜索 demo（独立，不依赖 langchain）。

对 data/langchain-docs 下的 markdown 做稀疏(词频)检索，只用 bm25s + 标准库。

用法:
    uv run python src/helpdesk/bm25.py "how to define a tool"
    uv run python src/helpdesk/bm25.py            # 交互模式
"""

import argparse
from pathlib import Path

import bm25s

DOCS_DIR = Path(__file__).resolve().parents[2] / "data" / "langchain-docs"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 5


def load_chunks(docs_dir: Path = DOCS_DIR) -> list[tuple[str, str]]:
    """读 markdown，按固定窗口简单切块，返回 [(来源, 文本)]。"""
    chunks: list[tuple[str, str]] = []
    step = CHUNK_SIZE - CHUNK_OVERLAP
    for path in sorted(docs_dir.rglob("*.md")):
        text = path.read_text(encoding="utf-8")
        source = str(path.relative_to(docs_dir))
        for start in range(0, len(text), step):
            piece = text[start : start + CHUNK_SIZE].strip()
            if piece:
                chunks.append((source, piece))
    return chunks


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
        preview = " ".join(text.split())[:90]
        print(f"  {rank}. {score:6.3f}  {source}")
        print(f"     {preview}...")


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
