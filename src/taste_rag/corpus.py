"""语料层：定位文档目录，提供两种切分方式。

只依赖标准库，供 bm25（固定窗口切块）和 vector_store（LangChain 切分器）共用。
"""

from pathlib import Path

DOCS_DIR = Path(__file__).resolve().parents[2] / "data" / "langchain-docs"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def load_chunks(docs_dir: Path = DOCS_DIR) -> list[tuple[str, str]]:
    """读 markdown，按固定窗口切块，返回 [(来源, 文本)]。"""
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
