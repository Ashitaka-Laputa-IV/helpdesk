"""会话标识：多用户 / 多线程的 thread_id 约定。

图本身不关心用户是谁，隔离全靠 checkpointer 的 thread_id；
拼法统一放这里，避免每个入口各写一套。
"""

DEFAULT_USER = "default"
DEFAULT_THREAD = "basic"  # 命令行默认会话名


def thread_key(user_id: str, thread_id: str) -> str:
    """拼出 checkpointer 的 thread_id，保证不同用户/会话互不串线。"""
    return f"{user_id}:{thread_id}"


def thread_config(user_id: str, thread_id: str) -> dict:
    """构造传给图的 config。"""
    return {"configurable": {"thread_id": thread_key(user_id, thread_id)}}
