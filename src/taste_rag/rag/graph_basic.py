"""最简 2-step RAG：向量召回 -> rerank -> 流式生成。

单轮问答、无对话历史、无路由判断，用来做基线对照。
索引与开库逻辑在 vector_store.py，检索原语在 retrieval.py。

用法:
    uv run python -m taste_rag.rag.graph_basic
"""

from collections.abc import Iterator
from pathlib import Path

from dotenv import load_dotenv
from langchain.messages import HumanMessage, SystemMessage

from taste_rag.core import runtime
from taste_rag.rag.common import PROMPT_WITH_DOCS
from taste_rag.core.retrieval import format_context, rerank_candidates, vector_candidates
from taste_rag.core.vector_store import INDEX_DIR, ensure_indexed, open_store

RECALL_K = 10  # 向量召回条数（重排前先放宽）
TOP_K = 5  # 重排后进入上下文的条数


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
        top = rerank_candidates(
            reranker, vector_candidates(retriever, question), question, TOP_K
        )
        messages = [
            SystemMessage(content=PROMPT_WITH_DOCS),
            HumanMessage(content=f"资料：\n{format_context(top)}\n\n问题：{question}"),
        ]
        for chunk in model.stream(messages):
            if chunk.text:
                yield chunk.text

    return answer


def main() -> None:
    runtime.silence_warnings()
    load_dotenv()
    answer = build_rag(
        runtime.load_model(), runtime.load_embeddings(), runtime.load_reranker()
    )
    runtime.run_cli(answer)


if __name__ == "__main__":
    main()
