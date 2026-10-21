"""把 chat 图包成 FastAPI 服务。

接口:
    POST /chat                              {user_id, thread_id, message} -> SSE 流式返回
    GET  /threads/{user_id}/{thread_id}     该会话的历史消息
    GET  /health

启动:
    uv run uvicorn taste_rag.api.server:app --reload
"""

import json
from collections.abc import Iterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import BaseModel

from taste_rag.chat.basic import DB_PATH, build_graph
from taste_rag.core import runtime
from taste_rag.core.session import thread_config


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时建图（checkpointer 贯穿整个服务生命周期），关闭时释放。"""
    runtime.silence_warnings()
    with SqliteSaver.from_conn_string(str(DB_PATH)) as checkpointer:
        app.state.graph = build_graph(runtime.load_model(), checkpointer)
        yield


app = FastAPI(title="taste-rag chat", lifespan=lifespan)


class ChatRequest(BaseModel):
    user_id: str
    thread_id: str
    message: str


def _sse(payload: dict) -> str:
    """把一条事件编码成 SSE 帧。"""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@app.post("/chat")
def chat(request: ChatRequest) -> StreamingResponse:
    """流式对话：逐 token 以 text/event-stream 返回。"""
    graph = app.state.graph
    config = thread_config(request.user_id, request.thread_id)

    def generate() -> Iterator[str]:
        stream = graph.stream_events(
            {"messages": [HumanMessage(content=request.message)]},
            config,
            version="v3",
        )
        for _, item in stream.interleave("messages"):
            if item.node != "chatbot":
                continue
            for token in item.text:
                yield _sse({"type": "token", "text": token})
        yield _sse({
            "type": "end",
            "user_id": request.user_id,
            "thread_id": request.thread_id,
        })

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/threads/{user_id}/{thread_id}")
def history(user_id: str, thread_id: str) -> dict:
    """返回该会话已有的消息。"""
    snapshot = app.state.graph.get_state(thread_config(user_id, thread_id))
    messages = (snapshot.values or {}).get("messages", [])
    return {
        "user_id": user_id,
        "thread_id": thread_id,
        "messages": [{"role": m.type, "content": m.text} for m in messages],
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
