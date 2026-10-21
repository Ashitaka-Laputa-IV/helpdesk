"""纯 create_agent 实现的多轮 chatbot（内存历史，不落盘）。

只提供构建函数，对话由测试脚本驱动，见 tests/compare_chatbots.py。
"""

from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

from taste_rag.core.tools import TOOLS

SYSTEM_PROMPT = "你是客服助手，回答要简洁；需要当前时间时调用工具获取，不要自己猜。"


def build_agent(model):
    """纯 create_agent 版本：图、工具循环、内存历史都由它自己管。"""
    return create_agent(
        model,
        tools=TOOLS,
        system_prompt=SYSTEM_PROMPT,
        # 内存 checkpointer：维持多轮历史，但进程结束就没了
        checkpointer=InMemorySaver(),
    )
