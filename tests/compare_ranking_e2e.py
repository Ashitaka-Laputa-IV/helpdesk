"""三种排序方案的端到端首字延迟对比：无排序 / rerank / RRF。

排序方式只影响检索后那一小段，但会直接体现在**首字延迟**上。
本脚本用同一张图（只切换 ranking），流式计时取第一个文本 token 的到达时间；
为抵消单次抖动，每种方案跑多轮并轮转执行顺序，取中位数。

用法:
    uv run python tests/compare_ranking_e2e.py
    uv run python tests/compare_ranking_e2e.py --repeat 3
"""

import argparse
import io
import os
import statistics
import time
from contextlib import redirect_stdout

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_reranker.rerankers import init_reranker
from langgraph.checkpoint.memory import InMemorySaver

from taste_rag.core.bm25 import build_index
from taste_rag.core.corpus import load_chunks
from taste_rag.core.vector_store import COLLECTION, INDEX_DIR
from taste_rag.core.runtime import DEFAULT_ROUTER_MAX_TOKENS as ROUTER_MAX_TOKENS, silence_serializer_warning
from taste_rag.rag.router import BM25_K_FUSION, SEMANTIC_K, build_graph

# 只测必检索的问题：非检索轮三方没有差别（都不排序）
QUESTIONS = [
    "How do I define a tool in LangChain?",
    "What is the basic flow of vector store retrieval?",
]
SCHEMES = ("none", "rerank", "rrf")
LABELS = {"none": "无排序", "rerank": "rerank", "rrf": "RRF"}


def invoke_streaming(graph, payload, config):
    """返回 (首字耗时, 总耗时)。"""
    started = time.perf_counter()
    ttft = None
    # stream_mode="messages" 每个 chunk 是 (AIMessageChunk, metadata) 元组；
    # 只认 chatbot 节点吐出的 token——否则会把 router 那次结构化调用的首个 chunk 当成首字
    for message, metadata in graph.stream(payload, config, stream_mode="messages"):
        if metadata.get("langgraph_node") != "chatbot":
            continue
        if ttft is None and getattr(message, "text", ""):
            ttft = time.perf_counter() - started
    return ttft, time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser(description="排序方案首字延迟对比")
    parser.add_argument("--repeat", type=int, default=3, help="每种方案重复轮数")
    args = parser.parse_args()

    load_dotenv()
    silence_serializer_warning()

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
    ).as_retriever(search_kwargs={"k": SEMANTIC_K})
    reranker = init_reranker(
        os.environ["RERANKER_MODEL"], base_url=os.environ["RERANKER_BASE_URL"]
    )
    model = init_chat_model(os.environ["MODEL"])
    router_model = init_chat_model(
        os.environ["ROUTER_MODEL"],
        base_url=os.environ["ROUTER_BASE_URL"],
        api_key=os.environ["OPENAI_API_KEY"],
        temperature=0,
        max_tokens=ROUTER_MAX_TOKENS,
    )

    graphs = {
        scheme: build_graph(
            model,
            router_model,
            bm25,
            chunks,
            retriever,
            InMemorySaver(),
            "llm",
            ranking=scheme,
            bm25_k=BM25_K_FUSION,
            reranker=reranker,
        )
        for scheme in SCHEMES
    }
    # 预热：避免首次建连、模型加载计入统计
    with redirect_stdout(io.StringIO()):
        invoke_streaming(
            graphs["rrf"],
            {"messages": [HumanMessage(content=QUESTIONS[0])]},
            {"configurable": {"thread_id": "warmup"}},
        )

    samples: dict[str, list[tuple[float, float]]] = {s: [] for s in SCHEMES}
    for round_index in range(args.repeat):
        # 每轮轮转执行顺序，抵消先后顺序带来的偏差
        order = SCHEMES[round_index % len(SCHEMES):] + SCHEMES[: round_index % len(SCHEMES)]
        for scheme in order:
            for q_index, question in enumerate(QUESTIONS):
                config = {"configurable": {"thread_id": f"{scheme}-{round_index}-{q_index}"}}
                with redirect_stdout(io.StringIO()):
                    ttft, total = invoke_streaming(
                        graphs[scheme], {"messages": [HumanMessage(content=question)]}, config
                    )
                samples[scheme].append((ttft or 0.0, total))

    print(f"\n===== 首字延迟（{args.repeat} 轮 × {len(QUESTIONS)} 个检索问题）=====")
    print(f"{'方案':<10}{'首字平均':>10}{'首字中位':>10}{'最快':>9}{'最慢':>9}{'总耗时中位':>12}")
    for scheme in SCHEMES:
        ttfts = [ttft for ttft, _ in samples[scheme]]
        totals = [total for _, total in samples[scheme]]
        print(
            f"{LABELS[scheme]:<10}"
            f"{statistics.mean(ttfts) * 1000:>9.0f}ms"
            f"{statistics.median(ttfts) * 1000:>9.0f}ms"
            f"{min(ttfts) * 1000:>8.0f}ms"
            f"{max(ttfts) * 1000:>8.0f}ms"
            f"{statistics.median(totals):>11.1f}s"
        )


if __name__ == "__main__":
    main()
