"""下载 LangChain 官方 Python 文档到 data/langchain-docs/。

只抓 oss/python 下的文档（LangChain / LangGraph 等库的使用知识），
排除 contributing，以及 langsmith、部署、评测等非库内容。

用法:
    uv run python data/download_docs.py
"""

import concurrent.futures as cf
import pathlib
import re
import urllib.request

INDEX_URL = (
    "https://docs.langchain.com/_llms/agent-development-lifecycle/build/python.txt"
)
BASE = "https://docs.langchain.com/"
PAGE_PATTERN = re.compile(r"https://docs\.langchain\.com/oss/python/[^)\s]+")
EXCLUDE = ("/contributing/",)
OUT_DIR = pathlib.Path(__file__).resolve().parent / "langchain-docs"


def collect_urls() -> list[str]:
    """从官方索引里取出 oss/python 文档 URL（去重、去锚点、剔除排除项）。"""
    with urllib.request.urlopen(INDEX_URL, timeout=30) as resp:
        index_text = resp.read().decode("utf-8")

    urls = set()
    for match in PAGE_PATTERN.findall(index_text):
        url = match.split("#", 1)[0].removesuffix(".md")
        if any(part in url for part in EXCLUDE):
            continue
        urls.add(url)
    return sorted(urls)


def download(url: str) -> int:
    """下载单个页面（markdown），路径镜像官方 URL 结构，返回字节数。"""
    rel = url.replace(BASE, "")
    dest = OUT_DIR / f"{rel}.md"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=30) as resp:
        content = resp.read()
    dest.write_bytes(content)
    return len(content)


def main() -> None:
    urls = collect_urls()
    succeeded, total, failed = 0, 0, []
    with cf.ThreadPoolExecutor(12) as executor:
        futures = {executor.submit(download, url): url for url in urls}
        for future in cf.as_completed(futures):
            url = futures[future]
            try:
                total += future.result()
                succeeded += 1
            except Exception as exc:  # noqa: BLE001
                failed.append((url, repr(exc)[:100]))

    print(f"成功 {succeeded}/{len(urls)}，共 {total / 1024 / 1024:.2f} MB -> {OUT_DIR}")
    for url, error in failed:
        print(f"  失败 {url} {error}")


if __name__ == "__main__":
    main()
