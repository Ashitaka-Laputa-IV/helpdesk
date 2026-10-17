
## graph图

```mermaid
graph TD
    Start([Start]) --> Router[Router]

    %% 路由分发逻辑
    Router -- "yes" --> BM25S["bm25s 40 chunks"]
    Router -- "yes" --> Semanti["semantic 20"]
    Router -- "No." --> LLM

    %% 检索与合并
    BM25S --> Combine[Combine 60]
    Semanti --> Combine
    
    %% 重排与大模型
    Combine --> Rerank[Rerank 5]
    Rerank --> LLM[LLM]
    LLM --> END([END])
```

## Router prompts

```text
当历史对话可以回答问题返回"yes"
当历史对话无法回答问题返回"no"
```

## LLM prompts

```text
根据"content"及其历史对话回答问题, 当"content"内容及其历史对话无法恢复"question"时, 则明确表示知识库内容未涵盖"question"的答案.

[question]:
{question}

[content]:
{content}
```

