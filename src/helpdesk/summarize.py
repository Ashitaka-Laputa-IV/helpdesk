"""对话历史的上下文压缩（sum 中间件）。

历史会随轮次无限增长，这里按 token 预算触发总结：把较早的对话压成一段摘要，
摘要以一条**带标记的 HumanMessage** 留在 messages 最前面，原文用 RemoveMessage 清掉，
最终只保留「摘要 + 最近 N 条原文」。

为什么用 HumanMessage 而不是 SystemMessage：在对话中间插入 system 消息，小模型的
chat template 往往不认；走 messages 也让状态只有一份事实来源，router/chatbot 直接读
state["messages"] 即可，不需要额外拼接摘要。
"""

import os

from langchain.agents.middleware.summarization import count_tokens_approximately
from langchain.messages import HumanMessage, RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES

# 触发阈值：历史超过该 token 数就压缩（粗略估算，4 字符≈1 token）
TRIGGER_TOKENS = int(os.environ.get("SUMMARY_TRIGGER_TOKENS", "1500"))
# 压缩后保留的最近消息条数（含当前这一条用户提问）
KEEP_MESSAGES = int(os.environ.get("SUMMARY_KEEP_MESSAGES", "6"))

# 摘要消息的标记，用来把它和真实对话区分开
SUMMARY_KIND = "summary"
SUMMARY_HEADER = "（以下是更早对话的摘要，不是用户新说的话）\n"

SUMMARY_PROMPT = (
    "你在压缩客服对话历史。把「新增对话」合并进「已有摘要」，输出更新后的摘要。\n"
    "要求：\n"
    "- 保留用户问过的问题，以及答案里的关键结论、具体数字、名称、配置项\n"
    "- 保留「这个问题已经回答过」的事实，便于后续判断是否要重复检索\n"
    "- 丢弃寒暄、重复提问的冗余内容和无关细节\n"
    "- 用简短条目，不编造，总长度不超过 300 字\n\n"
    "[已有摘要]\n{summary}\n\n[新增对话]\n{dialogue}"
)


def count_tokens(messages) -> int:
    """粗略估算消息占用的 token 数。"""
    return int(count_tokens_approximately(messages))


def is_summary(message) -> bool:
    """判断该消息是不是历史摘要。"""
    return message.additional_kwargs.get("kind") == SUMMARY_KIND


def split_history(messages) -> tuple[list, list]:
    """把历史拆成（已有摘要, 原文消息）。"""
    summaries = [m for m in messages if is_summary(m)]
    plain = [m for m in messages if not is_summary(m)]
    return summaries, plain


def safe_cutoff(messages, keep: int) -> int:
    """算出压缩的分界点：保留最后 keep 条。

    不能把 AI 的 tool_calls 和它后面的 ToolMessage 拆开，否则历史里会留下
    孤立的工具结果，部分模型会直接报错，因此遇到 tool 消息就往前退。
    """
    cut = max(0, len(messages) - keep)
    while 0 < cut < len(messages) and messages[cut].type == "tool":
        cut -= 1
    return cut


def format_dialogue(messages) -> str:
    """把消息渲染成单行文本，给总结模型用。"""
    lines = []
    for message in messages:
        text = getattr(message, "text", "") or ""
        if text:
            lines.append(f"{message.type}: {text}")
    return "\n".join(lines)


def build_summary_message(summary: str) -> HumanMessage:
    """把摘要包成一条带标记的 HumanMessage。"""
    return HumanMessage(
        content=f"{SUMMARY_HEADER}{summary}", additional_kwargs={"kind": SUMMARY_KIND}
    )


def summarize_if_needed(model, messages) -> tuple[str, list]:
    """超出预算时把较早的消息压成摘要。

    返回 (摘要文本, 需要保留的消息列表)。未触发或失败时返回 ("", [])，
    这样压缩失败也不会丢历史。
    """
    if count_tokens(messages) <= TRIGGER_TOKENS:
        return "", []

    summaries, plain = split_history(messages)
    cut = safe_cutoff(plain, KEEP_MESSAGES)
    if cut <= 0:
        return "", []

    old, remaining = plain[:cut], plain[cut:]
    if not old:
        return "", []

    previous = "\n".join(m.text for m in summaries if m.text)
    prompt = SUMMARY_PROMPT.format(
        summary=previous or "（无）", dialogue=format_dialogue(old)
    )
    try:
        reply = model.invoke(prompt)
    except Exception:  # noqa: BLE001 - 压缩失败不应阻塞对话
        return "", []
    new_summary = (getattr(reply, "text", "") or "").strip()
    if not new_summary:
        return "", []

    print(
        f"[summarize] 压缩 {len(old)} 条 -> 摘要 {len(new_summary)} 字，"
        f"保留 {len(remaining)} 条"
    )
    return new_summary, remaining


def compress_history(model, messages) -> list | None:
    """压缩历史，返回可以交给 messages 通道的更新列表；无需压缩时返回 None。

    用 REMOVE_ALL_MESSAGES 先清空再按顺序重放，保证摘要固定排在最近原文之前。
    """
    summary, remaining = summarize_if_needed(model, messages)
    if not summary:
        return None
    return [
        RemoveMessage(id=REMOVE_ALL_MESSAGES),
        build_summary_message(summary),
        *remaining,
    ]
