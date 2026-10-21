"""Agentic RAG：检索交给 create_agent 决定，而不是固定编排。

和 rag_router.py 的区别：
  - rag_router：router 节点先判「要不要检索」，再走固定的 召回 -> 合并 -> 重排 -> 生成。
  - agentic：只提供一个 search_knowledge_base 工具，何时检索、检索几次、要不要再查，
    全部由模型自己决定。

检索链路与固定编排保持一致：bm25(10) + 语义(15) 去重 -> 重排取前 5。
只提供构建函数，对比测试见 tests/compare_rag.py。
"""

from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from taste_rag.core.retrieval import (
    bm25_candidates,
    dedupe,
    format_context,
    rerank,
    sem_candidates,
)

BM25_K = 10
TOP_K = 5

# Agent 版没有固定的 router 节点，靠这段提示词约束「什么时候查、查到什么程度、查不到怎么说」
SYSTEM_PROMPT = """你是 LangChain / LangGraph 的技术客服，回答必须依据文档检索结果。

【什么时候调用工具 search_knowledge_base】
- 问题涉及 LangChain、LangGraph 的概念、API、类名、参数、配置方式时，先调用它，不要凭记忆作答。
- 一次检索结果不相关或信息不全时，换更准确的关键词再检索，最多 3 次。
- 寒暄（你好、谢谢、再见）、与文档无关的闲聊、或者对话历史里已经给出过答案时，不要调用工具，直接回答。

【怎么回答】
- 只使用工具返回的资料和对话历史，不要补充资料之外的内容。
- 资料里没有的信息，直接说明「资料中没有提到」，不要编造。
- 用中文简洁回答，必要时给出代码片段，并标注来源文件名。"""


def build_search_tool(bm25, chunks, retriever, reranker):
    """把混合召回 + 重排包装成一个工具，交给 agent 自主调用。"""

    @tool
    def search_knowledge_base(query: str) -> str:
        """在 LangChain / LangGraph 官方文档库中检索，返回最相关的几段原文。

        Args:
            query: 检索关键词，写清技术名词即可，不要写成完整长句。
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


def build_agent(llm, bm25, chunks, retriever, reranker):
    """构建 agentic RAG：工具自己调，流程不写死。"""
    tools = [build_search_tool(bm25, chunks, retriever, reranker)]
    return create_agent(
        llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=InMemorySaver(),
    )
