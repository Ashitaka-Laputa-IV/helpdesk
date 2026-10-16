"""最小 2-step RAG：加载 -> 切分 -> 嵌入 -> 检索 -> 重排 -> 生成。"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, SystemMessage
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_reranker.rerankers import init_reranker
from langchain_text_splitters import RecursiveCharacterTextSplitter

DOCS_DIR = Path(__file__).resolve().parents[2] / "data" / "langchain-docs"
INDEX_DIR = Path(__file__).resolve().parents[2] / ".chroma"
COLLECTION = "helpdesk"
SYSTEM_PROMPT = "你是客服助手，只依据给定资料回答问题；资料中没有的信息，直接说明不知道。"


def load_documents(data_dir: Path = DOCS_DIR) -> list[Document]:
    """把 data_dir 下的 markdown 读成 Document。"""
    return [
        Document(
            page_content=path.read_text(encoding="utf-8"),
            metadata={"source": str(path.relative_to(data_dir))},
        )
        for path in sorted(data_dir.rglob("*.md"))
    ]


def build_rag(
    model,
    embeddings,
    reranker,
    data_dir: Path = DOCS_DIR,
    index_dir: Path = INDEX_DIR,
):
    """构建 RAG 问答函数：检索 top-k，重排后取前 3 段作为上下文。

    索引持久化在 index_dir（Chroma）。首次运行会切分并写入，之后直接复用，不再重复嵌入。
    """
    store = Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings,
        persist_directory=str(index_dir),
    )
    if not store.get(limit=1)["ids"]:
        splits = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        ).split_documents(load_documents(data_dir))
        store.add_documents(splits)
        print(f"已索引 {len(splits)} 个分块到 {index_dir}")

    retriever = store.as_retriever(search_kwargs={"k": 10})

    def answer(question: str) -> str:
        candidates = retriever.invoke(question)
        top = reranker.compress_documents(candidates, query=question)[:3]
        context = "\n\n".join(
            f"[{doc.metadata['source']}] {doc.page_content}" for doc in top
        )
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"资料：\n{context}\n\n问题：{question}"),
        ]
        return model.invoke(messages).text

    return answer


def main() -> None:
    load_dotenv()
    model = init_chat_model(os.environ["MODEL"])
    embeddings = OpenAIEmbeddings(
        model=os.environ["EMBEDDING_MODEL"],
        base_url=os.environ["EMBEDDING_BASE_URL"],
        api_key=os.environ["OPENAI_API_KEY"],
        check_embedding_ctx_length=False,
    )
    reranker = init_reranker(
        os.environ["RERANKER_MODEL"], base_url=os.environ["RERANKER_BASE_URL"]
    )
    answer = build_rag(model, embeddings, reranker)

    print("输入问题开始问答，输入 exit / quit 结束。")
    while True:
        try:
            question = input("Q: ")
        except (EOFError, KeyboardInterrupt):
            break
        if question.strip().lower() in {"exit", "quit"}:
            break
        print(f"A: {answer(question)}")


if __name__ == "__main__":
    main()
