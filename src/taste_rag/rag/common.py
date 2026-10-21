"""各 RAG 实现共用的提示词、状态与节点。

summarize / router / chatbot 三个节点的逻辑在多个图里完全一样，
统一放这里；各 rag_* 只负责组装自己的图结构。
"""

import time
from typing import Literal

from langchain.messages import SystemMessage
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field

from taste_rag.core.summarize import compress_history

# 有资料 / 无资料两种回答姿势
PROMPT_WITH_DOCS = "你是客服助手，只依据给定资料回答问题；资料中没有的信息，直接说明不知道。"
PROMPT_PLAIN = "你是客服助手，请直接、简洁地回答用户。"

# Router 提示：直接把 state["messages"] 当对话输入，规则放在 system 里。
# 前提是 summarize 节点已把历史压短（默认保留最近 6 条）——历史一长，小模型会顺着对话作答而不是分类。
# 也不要问「是否需要检索外部资料」，模型会因"这我懂"一律答否；问「属于哪一类」才准。
PROMPT_ROUTER = (
    "你是路由器，只做分类，不要回答问题。判断最后一条用户消息属于哪种：\n"
    "- 对话历史里已给出过该问题的答案（含用户重复问同一句），或它是寒暄、闲聊、通用常识 -> chatbot\n"
    "- 历史完全没有相关信息，需要查阅 LangChain/LangGraph 技术文档 -> retrieve\n"
    "无法确定时选 retrieve。"
)


class RagState(MessagesState):
    """messages 是对话；context 是本轮检索到的资料。"""

    context: str


class RouterState(RagState):
    """在 RagState 之上记录路由决策与两路召回结果。"""

    search: bool
    bm25_docs: list[dict]
    semantic_docs: list[dict]


class RouteDecision(BaseModel):
    """Router 的输出：直接给出目标节点名，省掉文本解析。"""

    next: Literal["retrieve", "chatbot"] = Field(
        description="retrieve=需要检索知识库；chatbot=直接回答"
    )


def needs_retrieval(structured_router, messages) -> bool:
    """LLM 路由：把对话轮次直接作为输入，异常时保守返回 True（检索）。"""
    try:
        route = structured_router.invoke(
            [SystemMessage(content=PROMPT_ROUTER), *messages]
        )
    except Exception:  # noqa: BLE001 - 路由失败不应阻塞对话
        return True
    return route.next == "retrieve"


def make_summarize_node(model):
    """历史超预算时把较早的对话压成摘要，控制上下文长度。"""

    def summarize(state: RagState) -> dict:
        updates = compress_history(model, state["messages"])
        return {"messages": updates} if updates else {}

    return summarize


def make_router_node(structured_router, router_mode: str = "llm"):
    """判断本轮要不要检索，并清空上一轮的 context。

    router_mode:
      - "llm"：交给模型判断
      - "always"：一律检索（兜底或对比用）
    """

    def router(state: RouterState) -> dict:
        started = time.perf_counter()
        if router_mode == "llm":
            search_needed = needs_retrieval(structured_router, state["messages"])
        else:
            search_needed = True
        cost = time.perf_counter() - started
        print(
            f"[router] {'检索' if search_needed else '不检索'}"
            f"  {cost * 1000:.0f}ms  ({router_mode})"
        )
        return {"search": search_needed, "context": ""}

    return router


def make_answer_node(model):
    """有 context 就按资料作答，没有就普通回答。"""

    def answer(state: RagState) -> dict:
        context = state.get("context")
        prompt = (
            f"{PROMPT_WITH_DOCS}\n\n资料：\n{context}" if context else PROMPT_PLAIN
        )
        reply = model.invoke([SystemMessage(content=prompt), *state["messages"]])
        return {"messages": [reply]}

    return answer
