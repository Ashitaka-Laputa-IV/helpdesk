# data

RAG 语料目录。

- `download_docs.py`：下载脚本。
- `langchain-docs/`：下载结果，**不入库**（见 `.gitignore`），需要时重新下载。

## 语料内容

LangChain 官方 Python 文档（`https://docs.langchain.com/oss/python/`），只包含库使用相关的知识：

- `langchain/`、`langgraph/`、`deepagents/`、`integrations/`、`concepts/`、`migrate/`、`releases/` 等

已排除：`contributing/`，以及 langsmith / 部署 / 评测等非库内容。

源站提供干净的 markdown 接口（`{path}.md`），无需 HTML 解析。

## 使用

```bash
uv run python data/download_docs.py
```

下载约 180 个 markdown 文件（约 5 MB）到 `data/langchain-docs/`，目录结构镜像官方路径。

语料更新后需重建向量索引：删除项目根目录的 `.chroma/`，再按根目录 README 的「建向量索引」命令重新生成。

## 版权

文档版权归 LangChain 所有，仅供本地学习/测试使用。
