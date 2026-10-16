"""最简单的 chat 图：使用 langgraph + langchain。

图结构：START -> chatbot -> END
其中 chatbot 节点调用聊天模型，把模型的回复追加进 messages。
"""

import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage
from langgraph.graph import END, START, MessagesState, StateGraph


def build_graph(model):
    """构建并编译 chat 图。"""

    def chatbot(state: MessagesState) -> dict:
        return {"messages": [model.invoke(state["messages"])]}

    builder = StateGraph(MessagesState)
    builder.add_node("chatbot", chatbot)
    builder.add_edge(START, "chatbot")
    builder.add_edge("chatbot", END)
    return builder.compile()


def main() -> None:
    load_dotenv()
    model = init_chat_model(os.environ["MODEL"])
    graph = build_graph(model)

    messages = []
    print("输入内容开始对话，输入 exit / quit 结束。")
    while True:
        try:
            user_input = input("You: ")
        except (EOFError, KeyboardInterrupt):
            break
        if user_input.strip().lower() in {"exit", "quit"}:
            break

        messages.append(HumanMessage(content=user_input))
        messages = graph.invoke({"messages": messages})["messages"]
        print(f"AI: {messages[-1].text}")

if __name__ == '__main__':
    main()
