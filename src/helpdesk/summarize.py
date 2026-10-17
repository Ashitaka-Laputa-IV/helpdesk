"""对话历史的上下文压缩（sum 中间件）。

历史会随轮次无限增长，这里按 token 预算触发总结：把较早的对话压成一段摘要，
只保留最近若干条原文，摘要在图的 state.summary 字段里传递。

摘要刻意**不塞进 messages**，原因有两点：
  1. 在对话中间插入 SystemMessage，小模型的 chat template 往往不认；
  2. messages 保持"最近原文"，router 压平历史、chatbot 拼 system 提示都更简单。
"""

import os

from langchain.agents.middleware.summarization import count_tokens_approximately

# 触发阈值：历史超过该 token 数就压缩（粗略估算，4 字符≈1 token）
TRIGGER_TOKENS = int(os.environ.get("SUMMARY_TRIGGER_TOKENS", "1500"))
# 压缩后保留的最近消息条数（含当前这一条用户提问）
KEEP_MESSAGES = int(os.environ.get("SUMMARY_KEEP_MESSAGES", "6"))

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
    """把消息渲染成单行文本，给总结模型和 router 用。"""
    lines = []
    for message in messages:
        text = getattr(message, "text", "") or ""
        if text:
            lines.append(f"{message.type}: {text}")
    return "\n".join(lines)


def format_history(messages, summary: str = "") -> str:
    """router 用的历史文本：早期摘要 + 最近若干条原文。"""
    parts = []
    if summary:
        parts.append(f"（更早对话的摘要）\n{summary}")
    recent = format_dialogue(messages[:-1]) if messages else ""
    if recent:
        parts.append(recent)
    return "\n".join(parts) or "（无历史）"


def summarize_if_needed(model, summary: str, messages) -> tuple[str, list[str]]:
    """超出预算时把较早的消息压成摘要。

    返回 (新的摘要, 需要删除的消息 id 列表)。未触发或失败时返回原摘要、空列表，
    这样压缩失败也不会丢历史。
    """
    if count_tokens(messages) <= TRIGGER_TOKENS:
        return summary, []

    cut = safe_cutoff(messages, KEEP_MESSAGES)
    if cut <= 0:
        return summary, []

    old, remaining = messages[:cut], messages[cut:]
    removed = [m.id for m in old if m.id]
    if not removed:
        return summary, []

    prompt = SUMMARY_PROMPT.format(
        summary=summary or "（无）", dialogue=format_dialogue(old)
    )
    try:
        reply = model.invoke(prompt)
    except Exception:  # noqa: BLE001 - 压缩失败不应阻塞对话
        return summary, []
    new_summary = (getattr(reply, "text", "") or "").strip()
    if not new_summary:
        return summary, []

    print(
        f"[summarize] 压缩 {len(old)} 条 -> 摘要 {len(new_summary)} 字，"
        f"保留 {len(remaining)} 条"
    )
    return new_summary, removed
