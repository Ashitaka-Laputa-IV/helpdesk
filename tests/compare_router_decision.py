"""有检索判断 vs 无判断（总是检索）的对比。

同一张图、同一套 RRF 检索，只切 router_mode：

  - decision：让模型判断是否需要检索（llm）
  - always  ：一律检索，不做任何判断

分两阶段测量（本地服务流式不返回 usage，只能分开测）：
  1. 首字延迟：stream 取到首字即断开，--repeat 轮 × 4 问；
  2. token 消耗：非流式 invoke，callback 统计所有 LLM 调用（含 router），--token-repeat 轮。

用法:
    uv run python tests/compare_router_decision.py
    uv run python tests/compare_router_decision.py --repeat 6 --token-repeat 3
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
from langchain_core.callbacks import BaseCallbackHandler
from langchain_openai import OpenAIEmbeddings
from langchain_reranker.rerankers import init_reranker
from langgraph.checkpoint.memory import InMemorySaver

from taste_rag.bm25 import build_index, load_chunks
from taste_rag.rag import COLLECTION, INDEX_DIR
from taste_rag.router_chat import ROUTER_MAX_TOKENS, silence_serializer_warning
from taste_rag.router_chat_rrf import SEMANTIC_K, build_graph

# (问题, 是否需要检索)：混入寒暄与历史问题，才能看出"要不要判断"的差别
CORPUS = [
    ("Hello", False),
    ("How do I define a tool in LangChain?", True),
    ("What technical question did I just ask?", False),
    ("Thanks", False),
]
SCHEMES = {"decision": ("llm", "有判断"), "always": ("always", "总是检索")}


class TokenUsage(BaseCallbackHandler):
    """统计一次运行里所有 LLM 调用的 token 与次数（含 router 的结构化调用）。"""

    def __init__(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0
        self.calls = 0

    def on_llm_end(self, response, **kwargs) -> None:  # noqa: ANN001
        self.calls += 1
        for generations in response.generations:
            for generation in generations:
                usage = getattr(getattr(generation, "message", None), "usage_metadata", None)
                if usage:
                    self.input_tokens += usage.get("input_tokens", 0)
                    self.output_tokens += usage.get("output_tokens", 0)


def percentile(values: list[float], ratio: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * ratio))]


def measure_ttft(graph, question: str, thread: str) -> float | None:
    """流式取首字，拿到即关闭流。"""
    config = {"configurable": {"thread_id": thread}}
    started = time.perf_counter()
    stream = graph.stream(
        {"messages": [HumanMessage(content=question)]}, config, stream_mode="messages"
    )
    try:
        for message, metadata in stream:
            if metadata.get("langgraph_node") != "chatbot":
                continue
            if getattr(message, "text", ""):
                return time.perf_counter() - started
    finally:
        stream.close()
    return None


def measure_tokens(graph, question: str, thread: str) -> dict:
    """非流式跑完整一轮，返回 token 与调用次数。"""
    config = {"configurable": {"thread_id": thread}}
    counter = TokenUsage()
    with redirect_stdout(io.StringIO()):
        graph.invoke(
            {"messages": [HumanMessage(content=question)]},
            {**config, "callbacks": [counter]},
        )
    state = graph.get_state(config).values
    return {
        "input_tokens": counter.input_tokens,
        "output_tokens": counter.output_tokens,
        "calls": counter.calls,
        "searched": bool(state.get("context")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="检索判断 vs 总是检索")
    parser.add_argument("--repeat", type=int, default=6, help="首字测量轮数")
    parser.add_argument("--token-repeat", type=int, default=3, help="token 测量轮数")
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
        key: build_graph(
            model, router_model, bm25, chunks, retriever,
            InMemorySaver(), mode, ranking="rrf",
        )
        for key, (mode, _) in SCHEMES.items()
    }

    # 预热
    with redirect_stdout(io.StringIO()):
        measure_ttft(graphs["decision"], CORPUS[0][0], "warmup")

    print(f"首字：{args.repeat} 轮 × {len(CORPUS)} 问；Token：{args.token_repeat} 轮 × {len(CORPUS)} 问\n")

    ttft_rows: dict[str, list[dict]] = {}
    for key in SCHEMES:
        rows = []
        for round_index in range(args.repeat):
            for q_index, (question, need_search) in enumerate(CORPUS):
                ttft = measure_ttft(graphs[key], question, f"{key}-t-{round_index}-{q_index}")
                rows.append({"question": question, "need_search": need_search, "ttft": ttft or 0.0})
        ttft_rows[key] = rows

    token_rows: dict[str, list[dict]] = {}
    for key in SCHEMES:
        rows = []
        for round_index in range(args.token_repeat):
            for q_index, (question, need_search) in enumerate(CORPUS):
                stats = measure_tokens(
                    graphs[key], question, f"{key}-k-{round_index}-{q_index}"
                )
                rows.append({"question": question, **stats})
        token_rows[key] = rows

    print("===== 汇总 =====")
    print(
        f"{'方案':<10}{'首字中位':>10}{'首字P95':>10}"
        f"{'输入均值':>10}{'输入中位':>10}{'输出均值':>10}{'LLM调用/轮':>12}{'检索比例':>10}"
    )
    for key in SCHEMES:
        ttfts = [r["ttft"] for r in ttft_rows[key] if r["ttft"]]
        in_list = [r["input_tokens"] for r in token_rows[key]]
        in_tokens = statistics.mean(in_list)
        in_median = statistics.median(in_list)
        out_tokens = statistics.mean([r["output_tokens"] for r in token_rows[key]])
        calls = statistics.mean([r["calls"] for r in token_rows[key]])
        searched = sum(r["searched"] for r in token_rows[key])
        print(
            f"{SCHEMES[key][1]:<10}"
            f"{statistics.median(ttfts) * 1000:>9.0f}ms"
            f"{percentile(ttfts, 0.95) * 1000:>9.0f}ms"
            f"{in_tokens:>10.0f}"
            f"{in_median:>10.0f}"
            f"{out_tokens:>10.0f}"
            f"{calls:>12.1f}"
            f"{searched:>7}/{len(token_rows[key])}"
        )

    print("\n===== 分问题 =====")
    for question, need_search in CORPUS:
        print(f"  Q: {question}  (期望检索={'是' if need_search else '否'})")
        for key in SCHEMES:
            ttft_subset = [r for r in ttft_rows[key] if r["question"] == question and r["ttft"]]
            token_subset = [r for r in token_rows[key] if r["question"] == question]
            ttft = statistics.median([r["ttft"] for r in ttft_subset]) * 1000 if ttft_subset else 0
            in_tokens = statistics.mean([r["input_tokens"] for r in token_subset])
            out_tokens = statistics.mean([r["output_tokens"] for r in token_subset])
            searched = sum(r["searched"] for r in token_subset)
            print(
                f"     {SCHEMES[key][1]:<8} 首字 {ttft:6.0f}ms"
                f" | 输入 {in_tokens:5.0f} 输出 {out_tokens:5.0f}"
                f" | 检索 {searched}/{len(token_subset)}"
            )


if __name__ == "__main__":
    main()
