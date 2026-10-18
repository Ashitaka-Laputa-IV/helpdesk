"""create_agent 与 graph 编排结合的多轮 chatbot（内存历史，不落盘）。

图：START -> agent -> END，其中 agent 节点就是一个 create_agent 产物。
只提供构建函数，对话由测试脚本驱动，见 tests/compare_chatbots.py。
"""

from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph

from helpdesk import TOOLS
from helpdesk.agentic_chat import SYSTEM_PROMPT


def build_graph(model):
    """把 create_agent 产出的 agent 当成图节点，外面套一层 StateGraph。"""
    agent = create_agent(model, tools=TOOLS, system_prompt=SYSTEM_PROMPT)
    builder = StateGraph(MessagesState)
    builder.add_node("agent", agent)
    builder.add_edge(START, "agent")
    builder.add_edge("agent", END)
    return builder.compile(checkpointer=InMemorySaver())
