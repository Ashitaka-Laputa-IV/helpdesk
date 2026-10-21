"""把底层异常压成一句可读提示。

CLI 与 SSE 都不该把整段 traceback 甩给用户：这里沿异常链归类，
只保留最有信息量的一句。纯标准库，客户端也能直接复用。
"""

_MAX_LENGTH = 200

_CONNECTION_TYPES = {
    "APIConnectionError",
    "ConnectError",
    "ConnectionError",
    "ConnectionRefusedError",
    "ConnectTimeout",
    "NewConnectionError",
}
_TIMEOUT_TYPES = {
    "APITimeoutError",
    "ReadTimeout",
    "WriteTimeout",
    "TimeoutError",
    "TimeoutException",
}


def _chain(error: BaseException) -> list[BaseException]:
    """沿 __cause__ / __context__ 收集异常链，由外到内，避免循环。"""
    seen: set[int] = set()
    chain: list[BaseException] = []
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append(current)
        current = current.__cause__ or current.__context__
    return chain


def explain_error(error: BaseException) -> str:
    """返回一句面向用户的提示（不含堆栈）。"""
    chain = _chain(error)
    names = {type(item).__name__ for item in chain}
    text = " ".join(str(item) for item in chain).lower()

    if names & _CONNECTION_TYPES or "connection refused" in text:
        return (
            "连接依赖服务失败：请确认模型 / 嵌入 / 重排服务已启动，"
            "且 .env 中的地址配置正确。"
        )
    if names & _TIMEOUT_TYPES or "timed out" in text:
        return "请求超时：服务可能繁忙，可以稍后重试。"
    if "401" in text or "unauthorized" in text or "api key" in text:
        return "鉴权失败：请检查 OPENAI_API_KEY。"
    if "429" in text or "rate limit" in text:
        return "请求过于频繁：服务限流，请稍后重试。"
    if "404" in text or "not found" in text:
        return "接口或模型不存在：请检查 MODEL 名称与服务地址。"

    root = chain[-1]
    message = str(root).strip() or root.__class__.__name__
    return f"{root.__class__.__name__}: {message[:_MAX_LENGTH]}"
