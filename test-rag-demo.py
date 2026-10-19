"""rag-demo 的测试与效果统计。

不改动 rag-demo.py：通过 importlib 加载，用公开接口 stream()/state() 采集数据。

统计每轮：
  - 首字延迟（TTFT，取到首个 token 即记录）
  - 总耗时
  - 是否触发检索（看 state.context 是否非空）
  - 检索判定是否符合预期、答案是否命中关键词

用法:
    uv run python test-rag-demo.py
    uv run python test-rag-demo.py --repeat 2 --ranking rrf
"""

from __future__ import annotations

import argparse
import importlib.util
import statistics
import sys
import time
import warnings
from pathlib import Path

# (问题, 是否应该检索, 答案应命中的关键词之一)
CORPUS = [
    ("Hello", False, []),
    ("How do I define a tool in LangChain?", True, ["tool"]),
    ("What is the basic flow of vector store retrieval?", True, ["retriev", "vector"]),
    ("What technical question did I just ask?", False, ["tool", "vector"]),
    ("Thanks", False, []),
    ("How do I define a tool in LangChain?", False, ["tool"]),
]


def load_demo_module():
    """按路径加载 rag-demo.py（文件名带连字符，不能直接 import）。"""
    path = Path(__file__).with_name("rag-demo.py")
    spec = importlib.util.spec_from_file_location("rag_demo", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 {path}")
    module = importlib.util.module_from_spec(spec)
    # 必须先注册到 sys.modules：模块用了 from __future__ import annotations，
    # dataclass 解析字符串注解时会按模块名回查 sys.modules
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run_corpus(demo, round_index: int) -> list[dict]:
    """跑完整套语料，返回每轮记录。"""
    thread_id = f"test-{round_index}"
    rows = []
    for question, expect_search, keywords in CORPUS:
        started = time.perf_counter()
        first_token = None
        for _ in demo.stream(question, thread_id):
            if first_token is None:
                first_token = time.perf_counter() - started
        total = time.perf_counter() - started

        state = demo.state(thread_id)
        answer = state["messages"][-1].text
        searched = bool(state.get("context"))

        rows.append(
            {
                "round": round_index + 1,
                "question": question,
                "expect_search": expect_search,
                "searched": searched,
                "search_ok": searched == expect_search,
                "ttft": first_token or 0.0,
                "total": total,
                "answer": answer,
                "answer_ok": any(k in answer for k in keywords) if keywords else True,
            }
        )
    return rows


def report(rows: list[dict], ranking: str) -> None:
    print(f"\n===== 逐轮明细（ranking={ranking}）=====")
    print(f"{'轮':>3}{'首字':>9}{'总耗时':>9}{'检索':>7}{'判定':>5}{'关键词':>7}  问题")
    for row in rows:
        print(
            f"{row['round']:>3}"
            f"{row['ttft'] * 1000:>8.0f}ms"
            f"{row['total']:>8.1f}s"
            f"{'是' if row['searched'] else '否':>7}"
            f"{'OK' if row['search_ok'] else 'X':>5}"
            f"{'OK' if row['answer_ok'] else 'X':>7}"
            f"  {row['question'][:44]}"
        )
        print(f"     A: {row['answer'][:100].replace(chr(10), ' ')}")

    ttfts = [row["ttft"] for row in rows]
    totals = [row["total"] for row in rows]
    print(
        f"\n汇总: 首字中位 {statistics.median(ttfts) * 1000:.0f}ms"
        f" | 首字平均 {statistics.mean(ttfts) * 1000:.0f}ms"
        f" | 总耗时 {sum(totals):.1f}s"
        f" | 检索判定 {sum(r['search_ok'] for r in rows)}/{len(rows)}"
        f" | 关键词 {sum(r['answer_ok'] for r in rows)}/{len(rows)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="rag-demo 测试")
    parser.add_argument("--config", default="rag-demo.toml", help="配置文件路径")
    parser.add_argument("--repeat", type=int, default=2, help="整套语料重复轮数")
    parser.add_argument("--ranking", choices=["rrf", "rerank", "none"], help="覆盖排序策略")
    args = parser.parse_args()

    # with_structured_output 的 parsed 字段在流式序列化时会触发 Pydantic 告警
    warnings.filterwarnings("ignore", message="Pydantic serializer warnings")

    module = load_demo_module()
    settings = module.load_settings(Path(args.config).resolve())
    if args.ranking:
        from dataclasses import replace

        settings = replace(
            settings,
            retrieval=replace(settings.retrieval, ranking=args.ranking),
        )

    demo = module.RagDemo(settings)
    print(
        f"配置: ranking={settings.retrieval.ranking} | router={settings.retrieval.router_mode}"
        f" | bm25_k={settings.retrieval.bm25_k} | sem_k={settings.retrieval.semantic_k}"
        f" | rrf_k={settings.retrieval.rrf_k} | top_n={settings.retrieval.top_n}"
        f" | 语料分块 {len(demo._chunks)}"  # noqa: SLF001 - 仅用于展示规模
    )

    # 预热：排除首次建连
    demo.invoke("Hello", "warmup")

    rows: list[dict] = []
    for round_index in range(args.repeat):
        rows.extend(run_corpus(demo, round_index))

    report(rows, settings.retrieval.ranking)


if __name__ == "__main__":
    main()
