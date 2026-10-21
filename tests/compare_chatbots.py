"""串行对比两个多轮 chatbot：纯 create_agent vs LangGraph 手工编排。

同一模型、同一套语料；先跑完 A 的全部轮次，再跑完 B（串行执行，不并行）。
每轮记录耗时、消息条数、工具调用与校验结果，最后输出逐轮对比。

用法:
    uv run python tests/compare_chatbots.py
"""

import os
import time

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage

from taste_rag.agent_graph import build_graph
from taste_rag.agentic_chat import build_agent

# 8 轮语料：自我介绍 -> 稳定事实 -> 工具 -> 反复回忆 -> 归纳
CORPUS = [
    "你好，我叫李浩瑞",
    "我是一名后端工程师，主要写 Python",
    "现在几点？",
    "我叫什么名字？",
    "我是做什么工作的？",
    "我主要用什么编程语言？",
    "把刚才说的名字、职业和语言合成一句话",
    "现在几点？",
]

# 每轮校验：(答案里应出现的关键词, 应调用的工具)
CHECKS = [
    ([], None),
    ([], None),
    ([], "get_current_time"),
    (["李浩瑞"], None),
    (["后端", "工程师"], None),
    (["Python"], None),
    (["李浩瑞"], None),
    ([], "get_current_time"),
]


def run(name: str, graph, thread_id: str) -> list[dict]:
    """串行跑完整套语料，返回每轮记录。"""
    config = {"configurable": {"thread_id": thread_id}}
    rows = []
    seen = 0  # 上轮结束时的消息条数，用来只统计本轮新增的工具调用
    for index, question in enumerate(CORPUS):
        keywords, expected_tool = CHECKS[index]
        started = time.perf_counter()
        result = graph.invoke({"messages": [HumanMessage(content=question)]}, config)
        cost = time.perf_counter() - started

        messages = result["messages"]
        turn_messages = messages[seen:]
        seen = len(messages)
        tools = [
            call["name"]
            for message in turn_messages
            if message.type == "ai"
            for call in (message.tool_calls or [])
        ]
        answer = messages[-1].text

        ok = True
        if keywords:
            ok = any(word in answer for word in keywords)
        if expected_tool:
            ok = ok and expected_tool in tools

        rows.append(
            {
                "index": index + 1,
                "question": question,
                "answer": answer,
                "cost": cost,
                "messages": len(messages),
                "tools": tools,
                "ok": ok,
            }
        )
    return rows


def report(name: str, rows: list[dict]) -> None:
    print(f"\n===== {name} =====")
    for row in rows:
        tools = ",".join(row["tools"]) or "-"
        print(
            f"[{row['index']}] {row['cost'] * 1000:6.0f}ms  "
            f"消息{row['messages']:2d}  工具 {tools:16s} {'OK' if row['ok'] else 'X'}"
        )
        print(f"    Q: {row['question']}")
        print(f"    A: {row['answer']}")

    costs = [row["cost"] for row in rows]
    passed = sum(row["ok"] for row in rows)
    print(
        f"  汇总: 总计 {sum(costs):.2f}s | 平均 {sum(costs) / len(costs) * 1000:.0f}ms"
        f" | 最快 {min(costs) * 1000:.0f}ms | 最慢 {max(costs) * 1000:.0f}ms"
        f" | 校验 {passed}/{len(rows)}"
    )


def compare(left: list[dict], right: list[dict]) -> None:
    print("\n===== 逐轮耗时对比 =====")
    print("轮次 | A 纯 create_agent | B LangGraph 编排 | 差值")
    for a, b in zip(left, right):
        diff = (b["cost"] - a["cost"]) * 1000
        print(
            f"  {a['index']}  |      {a['cost'] * 1000:6.0f}ms     "
            f"|    {b['cost'] * 1000:6.0f}ms    | {diff:+.0f}ms"
        )

    total_a = sum(row["cost"] for row in left)
    total_b = sum(row["cost"] for row in right)
    print(
        f"  总计: A {total_a:.2f}s | B {total_b:.2f}s | 差值 {(total_b - total_a) * 1000:+.0f}ms"
    )
    same = sum(a["answer"] == b["answer"] for a, b in zip(left, right))
    print(f"  答案完全一致: {same}/{len(left)} 轮")


def main() -> None:
    load_dotenv()
    model = init_chat_model(os.environ["MODEL"])

    # 串行：A 跑完再跑 B，两个实现各自使用独立的 thread_id
    rows_a = run("A", build_agent(model), "cmp-a")
    rows_b = run("B", build_graph(model), "cmp-b")

    report("A. 纯 create_agent", rows_a)
    report("B. LangGraph 手工编排", rows_b)
    compare(rows_a, rows_b)


if __name__ == "__main__":
    main()
