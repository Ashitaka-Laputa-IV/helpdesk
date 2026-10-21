"""RAG 程序首字响应时间（TTFT）基准。

四种方案在同一批问题上测首字延迟：
  - none   ：无检索后排序
  - rerank ：cross-encoder 重排
  - rrf    ：名次倒数融合
  - agent  ：create_agent + 检索工具（模型自主决定何时检索）

计时口径：
  - 从 graph.stream 开始，到**回答节点**吐出第一个有内容的 token 为止；
  - 取到首字后立即关闭流，不等待整段生成，因此测量结果不含生成时间；
  - 每个样本使用独立 thread_id，避免历史影响。

用法:
    uv run python tests/benchmark_ttft.py
    uv run python tests/benchmark_ttft.py --repeat 10
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
from taste_rag.rag.agent import build_agent
from taste_rag.core.runtime import DEFAULT_ROUTER_MAX_TOKENS as ROUTER_MAX_TOKENS, silence_serializer_warning
from taste_rag.rag.router import BM25_K_FUSION, SEMANTIC_K, build_graph

QUESTIONS = [
    "How do I define a tool in LangChain?",
    "What is the basic flow of vector store retrieval?",
]
SCHEMES = ("none", "rerank", "rrf", "agent")
LABELS = {"none": "无排序", "rerank": "rerank", "rrf": "RRF", "agent": "Agentic"}
# 回答节点：固定编排是 chatbot，create_agent 内部是 model
ANSWER_NODES = {"chatbot", "model", "agent"}


def measure_ttft(graph, question: str, thread_id: str) -> float | None:
    """返回首字耗时（秒）；取到首字即关闭流，不计后续生成。"""
    started = time.perf_counter()
    stream = graph.stream(
        {"messages": [HumanMessage(content=question)]},
        {"configurable": {"thread_id": thread_id}},
        stream_mode="messages",
    )
    try:
        for message, metadata in stream:
            if metadata.get("langgraph_node") not in ANSWER_NODES:
                continue
            if getattr(message, "text", ""):
                return time.perf_counter() - started
    finally:
        stream.close()
    return None


def percentile(values: list[float], ratio: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * ratio))]


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG 首字响应时间基准")
    parser.add_argument("--repeat", type=int, default=10, help="每种方案重复轮数")
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
            model, router_model, bm25, chunks, retriever, InMemorySaver(),
            "llm", ranking=scheme, bm25_k=BM25_K_FUSION, reranker=reranker,
        )
        for scheme in ("none", "rerank", "rrf")
    }
    graphs["agent"] = build_agent(model, bm25, chunks, retriever, reranker)

    # 预热：排除首次建连与模型加载
    with redirect_stdout(io.StringIO()):
        measure_ttft(graphs["rrf"], QUESTIONS[0], "warmup")

    samples: dict[str, list[list[float]]] = {s: [[] for _ in QUESTIONS] for s in SCHEMES}
    for round_index in range(args.repeat):
        # 每轮轮转顺序，抵消先后偏差
        order = SCHEMES[round_index % len(SCHEMES):] + SCHEMES[: round_index % len(SCHEMES)]
        for scheme in order:
            for q_index, question in enumerate(QUESTIONS):
                with redirect_stdout(io.StringIO()):
                    ttft = measure_ttft(
                        graphs[scheme], question, f"{scheme}-{round_index}-{q_index}"
                    )
                if ttft:
                    samples[scheme][q_index].append(ttft)

    total_samples = sum(len(v) for lists in samples.values() for v in lists)
    print(f"首字响应时间基准：{args.repeat} 轮 × {len(QUESTIONS)} 问 × {len(SCHEMES)} 方案")
    print(f"有效样本 {total_samples} 个（取到首字即断开，不含生成时间）\n")

    print(f"{'方案':<10}{'样本':>6}{'平均':>9}{'中位':>9}{'P95':>9}{'最快':>9}{'最慢':>9}")
    for scheme in SCHEMES:
        flat = [t for per_question in samples[scheme] for t in per_question]
        print(
            f"{LABELS[scheme]:<10}{len(flat):>6}"
            f"{statistics.mean(flat) * 1000:>8.0f}ms"
            f"{statistics.median(flat) * 1000:>8.0f}ms"
            f"{percentile(flat, 0.95) * 1000:>8.0f}ms"
            f"{min(flat) * 1000:>8.0f}ms"
            f"{max(flat) * 1000:>8.0f}ms"
        )

    print("\n分问题（中位）")
    print(f"{'方案':<10}{'Q1 定义工具':>14}{'Q2 检索流程':>14}")
    for scheme in SCHEMES:
        q1 = samples[scheme][0]
        q2 = samples[scheme][1]
        print(
            f"{LABELS[scheme]:<10}"
            f"{statistics.median(q1) * 1000:>12.0f}ms"
            f"{statistics.median(q2) * 1000:>12.0f}ms"
        )


if __name__ == "__main__":
    main()
