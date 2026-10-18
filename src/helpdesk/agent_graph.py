"""LangGraph 手工编排版多轮 chatbot（内存历史，不落盘）。

和 agentic_chat.py 的区别：那边一行 create_agent 全包，这边用 init_chat_model
拿到的 llm 自己搭 ReAct 循环：

    START -> chatbot ->(有 tool_calls)-> tools -> chatbot -> END

只提供构建函数，对话由测试脚本驱动，见 tests/compare_chatbots.py。
"""

from langchain.messages import SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from helpdesk import TOOLS
from helpdesk.agentic_chat import SYSTEM_PROMPT


def build_graph(llm):
    """用 init_chat_model 得到的 llm 手工搭图。"""
    llm_with_tools = llm.bind_tools(TOOLS)

    def chatbot(state: MessagesState) -> dict:
        messages = [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]
        return {"messages": [llm_with_tools.invoke(messages)]}

    def should_continue(state: MessagesState):
        return "tools" if state["messages"][-1].tool_calls else END

    builder = StateGraph(MessagesState)
    builder.add_node("chatbot", chatbot)
    builder.add_node("tools", ToolNode(TOOLS))
    builder.add_edge(START, "chatbot")
    builder.add_conditional_edges("chatbot", should_continue, ["tools", END])
    builder.add_edge("tools", "chatbot")
    return builder.compile(checkpointer=InMemorySaver())
