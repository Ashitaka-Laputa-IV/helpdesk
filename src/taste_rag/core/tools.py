"""供 chat / agent 复用的工具集。

工具只依赖标准库或轻量库，任何图都可以直接 bind。
"""

from datetime import datetime

from langchain.tools import tool


@tool
def get_current_time() -> str:
    """获取当前的日期与时间。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


TOOLS = [get_current_time]
