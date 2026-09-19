import json
import time
import uuid

from typing import List, Optional

from fastapi import (
    FastAPI,
    Header,
    HTTPException,
)

from fastapi.responses import (
    StreamingResponse,
)

from pydantic import BaseModel

from src.observability import (
    run_traced_agent,
)


# =========================================================
# App
# =========================================================

app = FastAPI(
    title="Tuwaiq Technical Support Agent",
    version="1.0.0",
)

MODEL_ID = "tuwaiq-tech-support-agent"


# =========================================================
# Schemas
# =========================================================

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = MODEL_ID
    messages: List[ChatMessage]
    temperature: float = 0.2
    stream: bool = False


# =========================================================
# Health
# =========================================================

@app.get("/health")
def health():

    return {
        "status": "ok",
        "model": MODEL_ID,
    }


# =========================================================
# OpenAI-compatible Model List
# =========================================================

@app.get("/v1/models")
def models():

    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_ID,
                "object": "model",
                "owned_by": "JoudAlrubaish",
            }
        ],
    }


# =========================================================
# Streaming Helper
# =========================================================

def stream_response(
    request_id: str,
    model: str,
    answer: str,
):

    # First chunk
    first_chunk = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {
                    "role": "assistant",
                },
                "finish_reason": None,
            }
        ],
    }

    yield (
        f"data: {json.dumps(first_chunk)}\n\n"
    )

    # Answer chunk
    content_chunk = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {
                    "content": answer,
                },
                "finish_reason": None,
            }
        ],
    }

    yield (
        f"data: {json.dumps(content_chunk)}\n\n"
    )

    # Final chunk
    final_chunk = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": "stop",
            }
        ],
    }

    yield (
        f"data: {json.dumps(final_chunk)}\n\n"
    )

    yield "data: [DONE]\n\n"


# =========================================================
# OpenAI-compatible Chat Completion
# =========================================================

@app.post("/v1/chat/completions")
def chat_completions(
    req: ChatCompletionRequest,
    authorization: Optional[str] = Header(
        default=None
    ),
):

    user_messages = [
        message.content
        for message in req.messages
        if message.role == "user"
    ]

    if not user_messages:

        raise HTTPException(
            status_code=400,
            detail="No user message provided.",
        )

    request_id = (
        f"chatcmpl-"
        f"{uuid.uuid4().hex[:12]}"
    )

    started = time.perf_counter()

    try:

        result = run_traced_agent(
            user_messages[-1]
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )

    latency_ms = (
        time.perf_counter() - started
    ) * 1000

    answer = result.get(
        "answer",
        "Unable to produce an answer.",
    )


    # =====================================================
    # Streaming response for Open WebUI
    # =====================================================

    if req.stream:

        return StreamingResponse(
            stream_response(
                request_id=request_id,
                model=MODEL_ID,
                answer=answer,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )


    # =====================================================
    # Normal response
    # =====================================================

    return {
        "id": request_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_ID,

        "choices": [
            {
                "index": 0,

                "message": {
                    "role": "assistant",
                    "content": answer,
                },

                "finish_reason": "stop",
            }
        ],

        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },

        "system_metadata": {

            "route":
                result.get(
                    "route"
                ),

            "router_source":
                result.get(
                    "router_source"
                ),

            "intent":
                result.get(
                    "intent"
                ),

            "trace_id":
                result.get(
                    "trace_id"
                ),

            "router_latency_ms":
                result.get(
                    "router_latency_ms"
                ),

            "total_latency_ms":
                latency_ms,

            "escalated":
                result.get(
                    "escalated",
                    False,
                ),
        },
    }