"""语料层：定位文档目录，提供两种切分方式。

只依赖标准库，供 bm25（固定窗口切块）和 vector_store（LangChain 切分器）共用。
"""

from pathlib import Path

DOCS_DIR = Path(__file__).resolve().parents[3] / "data" / "langchain-docs"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def load_chunks(
    docs_dir: Path = DOCS_DIR,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[tuple[str, str]]:
    """读 markdown，按固定窗口切块，返回 [(来源, 文本)]。

    chunk_overlap 必须小于 chunk_size，否则步长非正、无法切分。
    """
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap 必须小于 chunk_size")

    chunks: list[tuple[str, str]] = []
    step = chunk_size - chunk_overlap
    for path in sorted(docs_dir.rglob("*.md")):
        text = path.read_text(encoding="utf-8")
        source = str(path.relative_to(docs_dir))
        for start in range(0, len(text), step):
            piece = text[start : start + chunk_size].strip()
            if piece:
                chunks.append((source, piece))
    return chunks
