"""LangGraph 手工编排版多轮 chatbot（内存历史，不落盘）。

和 chat_create_agent.py 的区别：那边一行 create_agent 全包，这边用 init_chat_model
拿到的 llm 自己搭 ReAct 循环。图本身复用 chat.build_graph，只在外面补上人设与内存 checkpointer。

只提供构建函数，对话由测试脚本驱动，见 tests/compare_chatbots.py。
"""

from langgraph.checkpoint.memory import InMemorySaver

from taste_rag.chat import build_graph as build_chat_graph
from taste_rag.chat_create_agent import SYSTEM_PROMPT


def build_graph(llm):
    """用 init_chat_model 得到的 llm 手工搭图（内存历史）。"""
    return build_chat_graph(llm, InMemorySaver(), SYSTEM_PROMPT)
