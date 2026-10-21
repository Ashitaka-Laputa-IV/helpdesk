"""最简 2-step RAG：向量召回 -> rerank -> 流式生成。

单轮问答、无对话历史、无路由判断，用来做基线对照。
索引与开库逻辑在 vector_store.py。

用法:
    uv run python src/taste_rag/rag_basic.py
"""

import os
from collections.abc import Iterator
from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, SystemMessage
from langchain_openai import OpenAIEmbeddings
from langchain_reranker.rerankers import init_reranker

from taste_rag.vector_store import INDEX_DIR, ensure_indexed, open_store

RECALL_K = 10  # 向量召回条数（重排前先放宽）
TOP_K = 5  # 重排后进入上下文的条数
SYSTEM_PROMPT = "你是客服助手，只依据给定资料回答问题；资料中没有的信息，直接说明不知道。"


def build_rag(
    model,
    embeddings,
    reranker,
    index_dir: Path = INDEX_DIR,
):
    """构建 RAG 问答函数：向量召回 RECALL_K 条，重排后取前 TOP_K 段，流式生成。

    索引持久化在 index_dir（Chroma），首次运行会切分并写入，之后直接复用。
    """
    store = open_store(embeddings, index_dir)
    indexed = ensure_indexed(store)
    if indexed:
        print(f"已索引 {indexed} 个分块到 {index_dir}")

    retriever = store.as_retriever(search_kwargs={"k": RECALL_K})

    def answer(question: str) -> Iterator[str]:
        """检索 + 重排后，逐 token 产出回答。"""
        candidates = retriever.invoke(question)
        top = reranker.compress_documents(candidates, query=question)[:TOP_K]
        context = "\n\n".join(
            f"[{doc.metadata['source']}] {doc.page_content}" for doc in top
        )
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"资料：\n{context}\n\n问题：{question}"),
        ]
        for chunk in model.stream(messages):
            if chunk.text:
                yield chunk.text

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

        print("A: ", end="", flush=True)
        for token in answer(question):
            print(token, end="", flush=True)
        print()


if __name__ == "__main__":
    main()
