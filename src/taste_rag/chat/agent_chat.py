"""纯 create_agent 实现的多轮 chatbot。

两种用法：
- build_agent(model)：默认内存历史，供对比测试（tests/compare_chatbots.py）使用
- main()：命令行入口，用 SqliteSaver 把历史落盘到 .db/agent_chat.db

图、工具循环、消息合并都由 create_agent 负责，本模块只提供提示词与外壳。
"""

import argparse

from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

from taste_rag.core import runtime
from taste_rag.core.paths import db_path
from taste_rag.core.session import DEFAULT_THREAD, DEFAULT_USER, thread_config
from taste_rag.core.tools import TOOLS

DB_PATH = db_path("agent_chat.db")
SYSTEM_PROMPT = "你是客服助手，回答要简洁；需要当前时间时调用工具获取，不要自己猜。"


def build_agent(model, checkpointer=None):
    """纯 create_agent 版本：图、工具循环和历史都由它自己管。

    checkpointer 为 None 时用内存历史（进程结束即丢）；main() 传 SqliteSaver 落盘。
    """
    return create_agent(
        model,
        tools=TOOLS,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer if checkpointer is not None else InMemorySaver(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="create_agent 版命令行 chat")
    parser.add_argument("--user", default=DEFAULT_USER, help="用户标识")
    parser.add_argument("--thread", default=DEFAULT_THREAD, help="会话标识")
    args = parser.parse_args()

    runtime.silence_warnings()
    config = thread_config(args.user, args.thread)

    # 历史存 .db/agent_chat.db，按 (user, thread) 隔离；create_agent 自己接上历史
    with SqliteSaver.from_conn_string(str(DB_PATH)) as checkpointer:
        agent = build_agent(runtime.load_model(), checkpointer)
        runtime.run_cli(
            # create_agent 内部的模型节点叫 model（不是 chatbot）
            lambda question: runtime.stream_answer(agent, config, question, node="model"),
            prompt="You: ",
            reply_prefix="AI: ",
            intro=f"[{args.user}/{args.thread}] 输入内容开始对话，输入 exit / quit 结束。",
        )


if __name__ == "__main__":
    main()
