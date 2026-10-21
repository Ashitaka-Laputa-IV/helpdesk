"""Chroma 向量库：文档加载、开库、按需建索引。

只做基础设施；检索后的问答流程见 rag/ 下的各实现。
"""

from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from taste_rag.core.corpus import CHUNK_OVERLAP, CHUNK_SIZE, DOCS_DIR

INDEX_DIR = Path(__file__).resolve().parents[3] / ".chroma"
COLLECTION = "helpdesk"  # 已持久化的索引名，改它会读不到现有向量库
BATCH_SIZE = 1000  # Chroma 单次写入有上限（约 5461 条），分批写入


def load_documents(data_dir: Path = DOCS_DIR) -> list[Document]:
    """把 data_dir 下的 markdown 读成 Document。"""
    return [
        Document(
            page_content=path.read_text(encoding="utf-8"),
            metadata={"source": str(path.relative_to(data_dir))},
        )
        for path in sorted(data_dir.rglob("*.md"))
    ]


def open_store(embeddings, index_dir: Path = INDEX_DIR) -> Chroma:
    """打开（或新建）持久化的 Chroma 集合。"""
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings,
        persist_directory=str(index_dir),
    )


def ensure_indexed(store: Chroma, data_dir: Path = DOCS_DIR) -> int:
    """集合为空时切分并写入，返回本次写入的分块数；已有索引返回 0。"""
    if store.get(limit=1)["ids"]:
        return 0
    splits = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    ).split_documents(load_documents(data_dir))
    for start in range(0, len(splits), BATCH_SIZE):
        store.add_documents(splits[start : start + BATCH_SIZE])
    return len(splits)
