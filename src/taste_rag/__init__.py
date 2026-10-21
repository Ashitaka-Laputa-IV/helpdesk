"""taste-rag：LangChain / LangGraph 的 RAG 实验集。

目录结构：
    core/   检索与运行时基础设施（语料、索引、召回、压缩、依赖装配）
    chat/   基础多轮 chat 与两个对照实现
    rag/    各类 RAG 实现与共用节点
    api/    FastAPI 服务与命令行客户端

这里只导出最基础的一组符号，方便 `from taste_rag import thread_config`；
其余按子包导入，例如 `from taste_rag.rag.router import build_graph`。
"""

from taste_rag.chat.basic import (
    DB_PATH,
    DEFAULT_THREAD,
    DEFAULT_USER,
    TOOLS,
    build_graph,
    get_current_time,
    load_model,
    main,
    silence_beta_warning,
    thread_config,
    thread_key,
)

__all__ = [
    "DB_PATH",
    "DEFAULT_THREAD",
    "DEFAULT_USER",
    "TOOLS",
    "build_graph",
    "get_current_time",
    "load_model",
    "main",
    "silence_beta_warning",
    "thread_config",
    "thread_key",
]
