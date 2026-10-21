"""构建检索评测集：标题级（简单档）+ 改写级（困难档）。

- 标题档：直接用文档 H1 标题当查询，正解为该文档。存在字面重合，偏简单。
- 改写档：取文档中段的一段内容，让本地模型生成一句"用户会怎么问"，正解仍是该文档。
  生成时不给出标题，并要求口语化，避免直接抄原文术语。

输出 data/eval/queries.json，结构：
    [{"tier": "title"|"paraphrase", "query": "...", "target": "oss/.../x.md", "title": "..."}]

用法:
    uv run python tests/build_eval_set.py --docs 30
"""

import argparse
import json
import os
import random
import re
from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

from taste_rag.corpus import DOCS_DIR, load_chunks

SEED = 42
OUT_PATH = Path(__file__).resolve().parents[1] / "data" / "eval" / "queries.json"

PROMPT = """Below is a technical documentation page: its title and one excerpt from it.
Write ONE question that a real user would ask, which this excerpt answers.

[Title]
{title}

[Excerpt]
{chunk}

Rules:
- Ask in English, natural/casual phrasing like in a support chat
- Stay on the topic of the title, do not drift into unrelated details
- Do not copy the title or sentences verbatim
- Output only the question, no explanation"""


def collect_docs() -> dict[str, dict]:
    """收集标题唯一的文档：{来源: {title, text}}。"""
    docs = {}
    for path in sorted(DOCS_DIR.rglob("*.md")):
        text = path.read_text(encoding="utf-8")
        match = re.search(r"^#\s+(.+)$", text, re.M)
        if match:
            docs[str(path.relative_to(DOCS_DIR))] = {
                "title": match.group(1).strip(),
                "text": text,
            }
    return docs


def build(doc_count: int) -> list[dict]:
    docs = collect_docs()
    titles = {}
    for source, info in docs.items():
        titles.setdefault(info["title"], []).append(source)

    # 标题唯一、且标题有多个词，避免 "Errors" 这类过泛的查询
    candidates = [
        source
        for title, sources in titles.items()
        if len(sources) == 1 and len(title.split()) >= 2
        for source in sources
    ]
    random.Random(SEED).shuffle(candidates)
    selected = candidates[:doc_count]

    # 按来源取中段分块，给改写档提供上下文
    chunks_by_source: dict[str, list[str]] = {}
    for source, text in load_chunks():
        chunks_by_source.setdefault(source, []).append(text)

    load_dotenv()
    model = init_chat_model(os.environ["MODEL"])

    rows = []
    for source in selected:
        info = docs[source]
        rows.append(
            {
                "tier": "title",
                "query": info["title"],
                "target": source,
                "title": info["title"],
            }
        )

        pieces = chunks_by_source.get(source) or [info["text"]]
        chunk = pieces[len(pieces) // 2][:1200]
        try:
            reply = model.invoke(PROMPT.format(title=info["title"], chunk=chunk))
        except Exception as exc:  # noqa: BLE001 - 生成失败就跳过该条
            print(f"跳过 {source}: {type(exc).__name__}")
            continue

        question = " ".join((getattr(reply, "text", "") or "").split())
        question = question.strip("\"'“”「」")
        if not (6 <= len(question) <= 160):
            print(f"跳过 {source}: 问题长度异常 {len(question)}")
            continue
        rows.append(
            {
                "tier": "paraphrase",
                "query": question,
                "target": source,
                "title": info["title"],
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="构建检索评测集")
    parser.add_argument("--docs", type=int, default=30, help="参与构建的文档数")
    args = parser.parse_args()

    rows = build(args.docs)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    paraphrase = [r for r in rows if r["tier"] == "paraphrase"]
    print(f"写入 {OUT_PATH}：共 {len(rows)} 条")
    print(f"  标题档 {len(rows) - len(paraphrase)} 条 | 改写档 {len(paraphrase)} 条")
    for row in paraphrase[:5]:
        print(f"  [{row['target']}] {row['query']}")


if __name__ == "__main__":
    main()
