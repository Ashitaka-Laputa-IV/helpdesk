"""Agentic RAG：检索交给 create_agent 决定，而不是固定编排。

和 rag/router.py 的区别：
  - rag/router：router 节点先判「要不要检索」，再走固定的 召回 -> 合并 -> 重排 -> 生成。
  - agentic：只提供一个 search_knowledge_base 工具，何时检索、检索几次、要不要再查，
    全部由模型自己决定。

检索链路与固定编排保持一致：bm25(10) + 语义(15) 去重 -> 重排取前 5。

build_agent 供对比测试使用（tests/compare_rag.py，默认内存历史）；
main() 是命令行入口，历史落盘到 .agentic_rag.db。

用法:
    uv run python -m taste_rag.rag.agentic
"""

from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

from taste_rag.core import runtime
from taste_rag.core.retrieval import (
    bm25_candidates,
    dedupe,
    format_context,
    rerank,
    sem_candidates,
)
from taste_rag.core.session import thread_config

DB_PATH = Path(__file__).resolve().parents[3] / ".agentic_rag.db"
BM25_K = 20
SEMANTIC_K = 15  # 与 rag/router.py 保持一致
TOP_K = 10

# Agent 版没有固定的 router 节点，靠这段提示词约束「什么时候查、查到什么程度、查不到怎么说」
SYSTEM_PROMPT = """你是一名技术客服，回答必须依据文档检索结果。

【什么时候调用工具 search_knowledge_base】
- 任何技术问题都先调用它：包括 LangChain / LangGraph 的概念、API、类名、参数、配置、用法，
  也包括没提框架名但属于技术范畴的问题（例如「中间件怎么自定义」「什么是 middleware」）。
  不要凭记忆作答，也不要先判断「资料里有没有」再决定查不查。
- 检索词一律用英文：文档库是英文的，中文问句要先提炼成英文技术名词再查。
- 一次检索结果不相关或信息不全时，换更准确的英文关键词再检索，最多 3 次。
- 只有以下情况可以不调用：寒暄（你好、谢谢、再见）、与文档无关的闲聊、
  或者对话历史里已经给出过答案。

【怎么回答】
- 只使用工具返回的资料和对话历史，不要补充资料之外的内容。
- 「资料中没有提到」只能在调用过工具、且返回的资料确实不包含答案时使用；
  没有调用工具时，禁止用这句话作答。
- 用中文简洁回答，必要时给出代码片段，并标注来源文件名。"""


def build_search_tool(bm25, chunks, retriever, reranker):
    """把混合召回 + 重排包装成一个工具，交给 agent 自主调用。"""

    @tool
    def search_knowledge_base(query: str) -> str:
        """在 LangChain / LangGraph 官方文档库（英文语料）中检索，返回最相关的几段原文。

        Args:
            query: 必须是英文关键词，用文档里的技术名词（如 "create_agent"、"interrupt"），
                不要写整句中文或完整长句。
        """
        # 1) 两路召回 -> 按 (来源, 原文) 去重
        candidates = dedupe(
            [
                bm25_candidates(bm25, chunks, query, BM25_K),
                sem_candidates(retriever, query),
            ]
        )

        # 2) 重排取前 TOP_K
        ranked = rerank(reranker, candidates, query, TOP_K)
        if not ranked:
            return "没有检索到相关内容。"
        return format_context(ranked)

    return search_knowledge_base


def build_agent(llm, bm25, chunks, retriever, reranker, checkpointer=None):
    """构建 agentic RAG：工具自己调，流程不写死。

    checkpointer 为 None 时用内存历史（测试用）；main() 传 SqliteSaver 落盘。
    """
    tools = [build_search_tool(bm25, chunks, retriever, reranker)]
    return create_agent(
        llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer if checkpointer is not None else InMemorySaver(),
    )


def main() -> None:
    runtime.silence_warnings()
    load_dotenv()

    stack = runtime.build_retrieval_stack(SEMANTIC_K, with_reranker=True)
    config = thread_config()

    # 历史与工具调用轨迹一起落盘，进程重启后还能接着问
    with SqliteSaver.from_conn_string(str(DB_PATH)) as checkpointer:
        agent = build_agent(
            runtime.load_model(),
            stack.bm25,
            stack.chunks,
            stack.retriever,
            stack.reranker,
            checkpointer,
        )
        runtime.run_cli(
            # create_agent 内部的模型节点叫 model
            lambda question: runtime.stream_answer(agent, config, question, node="model")
        )


if __name__ == "__main__":
    main()
