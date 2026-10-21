"""串行对比两个 RAG 实现：固定编排（rag_router）vs Agentic（rag_agent）。

同一模型、同一套语料；A 先跑完全部轮次，B 再跑一遍（串行执行，不并行）。
每轮记录：耗时、是否触发检索、答案、关键词校验。

用法:
    uv run python tests/compare_rag.py
"""

import io
import os
import time
from contextlib import redirect_stdout

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_reranker.rerankers import init_reranker
from langgraph.checkpoint.memory import InMemorySaver

from taste_rag.bm25 import build_index
from taste_rag.corpus import load_chunks
from taste_rag.vector_store import COLLECTION, INDEX_DIR
from taste_rag.rag_agent import build_agent
from taste_rag.rag_router import SEMANTIC_K, build_graph
from taste_rag.runtime import DEFAULT_ROUTER_MAX_TOKENS as ROUTER_MAX_TOKENS

# (问题, 是否应该检索, 答案应包含的关键词)
# 语料是英文文档，查询也必须用英文，否则测到的是语种不匹配而不是检索能力
CORPUS = [
    ("Hello", False, []),
    ("How do I define a tool in LangChain?", True, ["tool"]),
    ("What is the basic flow of vector store retrieval?", True, ["retriev", "vector"]),
    ("What technical question did I just ask?", False, ["tool", "vector"]),
    ("Thanks", False, []),
    ("How do I define a tool in LangChain?", False, ["tool"]),
]


def invoke_streaming(graph, payload, config):
    """流式跑一轮，返回 (首字耗时, 总耗时, 最终状态)。

    TTFT 取第一个带文本的 chunk，只计到"开始吐字"为止；
    后续生成过程再长也不计入，这更贴近用户感知的响应时间。
    """
    started = time.perf_counter()
    ttft = None
    final = None
    for mode, chunk in graph.stream(
        payload, config, stream_mode=["messages", "values"]
    ):
        if mode == "messages":
            message, metadata = chunk
            # 只认"回答节点"吐出的 token，否则会把 router / 工具决策那次调用的
            # 首个 chunk 误当成首字（A 是 chatbot 节点，create_agent 里是 model）
            if metadata.get("langgraph_node") not in {"chatbot", "model", "agent"}:
                continue
            if ttft is None and getattr(message, "text", ""):
                ttft = time.perf_counter() - started
        else:
            final = chunk
    return ttft, time.perf_counter() - started, final


def run(name: str, graph, is_agent: bool, thread_id: str) -> list[dict]:
    """串行跑完整套语料，返回每轮记录。"""
    config = {"configurable": {"thread_id": thread_id}}
    rows = []
    seen = 0  # 上轮结束时的消息条数，用来只看本轮新增内容
    for index, (question, expect_search, keywords) in enumerate(CORPUS):
        # rag_router 的节点会打印决策日志，这里屏蔽掉以免干扰报告
        with redirect_stdout(io.StringIO()):
            ttft, cost, result = invoke_streaming(
                graph, {"messages": [HumanMessage(content=question)]}, config
            )

        messages = result["messages"]
        new_messages = messages[seen:]
        seen = len(messages)

        if is_agent:
            # agent 版：本轮有没有调用检索工具
            tools = [
                call["name"]
                for message in new_messages
                if message.type == "ai"
                for call in (message.tool_calls or [])
            ]
            searched = bool(tools)
        else:
            # 固定编排：router 判定检索时才会填充 context
            searched = bool(result.get("context"))
            tools = ["search"] if searched else []

        answer = messages[-1].text
        rows.append(
            {
                "index": index + 1,
                "question": question,
                "expect_search": expect_search,
                "searched": searched,
                "search_ok": searched == expect_search,
                "answer": answer,
                "answer_ok": any(word in answer for word in keywords) if keywords else True,
                "ttft": ttft,
                "cost": cost,
                "messages": len(messages),
                "tools": tools,
            }
        )
    return rows


def report(name: str, rows: list[dict]) -> None:
    print(f"\n===== {name} =====")
    for row in rows:
        mark = "OK" if row["search_ok"] else "X"
        ttft = f"{row['ttft'] * 1000:6.0f}ms" if row["ttft"] else "     -"
        print(
            f"[{row['index']}] 首字 {ttft}  总 {row['cost'] * 1000:6.0f}ms  "
            f"检索={'是' if row['searched'] else '否'} "
            f"(期望{'是' if row['expect_search'] else '否'}) {mark}"
        )
        print(f"    Q: {row['question']}")
        print(f"    A: {row['answer'][:160].replace(chr(10), ' ')}")

    costs = [row["cost"] for row in rows]
    ttfts = [row["ttft"] for row in rows if row["ttft"]]
    search_ok = sum(row["search_ok"] for row in rows)
    answer_ok = sum(row["answer_ok"] for row in rows)
    print(
        f"  汇总: 首字平均 {sum(ttfts) / len(ttfts) * 1000:.0f}ms"
        f" | 总耗时 {sum(costs):.2f}s"
        f" | 检索判定 {search_ok}/{len(rows)} | 答案关键词 {answer_ok}/{len(rows)}"
    )


def compare(left: list[dict], right: list[dict]) -> None:
    print("\n===== 逐轮对比（首字耗时，括号内为总耗时）=====")
    print("轮次 | 期望检索 | A 固定编排            | B Agentic")
    for a, b in zip(left, right):
        expect = "是" if a["expect_search"] else "否"

        def cell(row: dict) -> str:
            ttft = f"{row['ttft'] * 1000:.0f}ms" if row["ttft"] else "-"
            return (
                f"{'检索' if row['searched'] else '不检索':6s} "
                f"{ttft:>7s} ({row['cost'] * 1000:.0f}ms)"
            )

        print(f"  {a['index']}  |    {expect}     | {cell(a):22s} | {cell(b)}")

    ttft_a = [r["ttft"] for r in left if r["ttft"]]
    ttft_b = [r["ttft"] for r in right if r["ttft"]]
    print(
        f"  首字平均: A {sum(ttft_a) / len(ttft_a) * 1000:.0f}ms"
        f" | B {sum(ttft_b) / len(ttft_b) * 1000:.0f}ms"
    )
    total_a = sum(row["cost"] for row in left)
    total_b = sum(row["cost"] for row in right)
    print(
        f"  总耗时: A {total_a:.2f}s | B {total_b:.2f}s | 差值 {(total_b - total_a) * 1000:+.0f}ms"
    )
    print(
        f"  检索判定正确: A {sum(r['search_ok'] for r in left)}/{len(left)}"
        f" | B {sum(r['search_ok'] for r in right)}/{len(right)}"
    )


def main() -> None:
    load_dotenv()

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
    print(f"语料分块 {len(chunks)} 条")

    model = init_chat_model(os.environ["MODEL"])
    router_model = init_chat_model(
        os.environ["ROUTER_MODEL"],
        base_url=os.environ["ROUTER_BASE_URL"],
        api_key=os.environ["OPENAI_API_KEY"],
        temperature=0,
        max_tokens=ROUTER_MAX_TOKENS,
    )

    # 串行：A 跑完再跑 B；各自独立的 thread 与内存 checkpointer，互不污染
    graph_a = build_graph(
        model,
        router_model,
        bm25,
        chunks,
        retriever,
        InMemorySaver(),
        "llm",
        reranker=reranker,
    )
    rows_a = run("A", graph_a, False, "rag-a")

    graph_b = build_agent(model, bm25, chunks, retriever, reranker)
    rows_b = run("B", graph_b, True, "rag-b")

    report("A. 固定编排（router -> 并行召回 -> 重排 -> 生成）", rows_a)
    report("B. Agentic（create_agent + 检索工具）", rows_b)
    compare(rows_a, rows_b)


if __name__ == "__main__":
    main()
