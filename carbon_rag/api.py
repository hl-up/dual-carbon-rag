"""Bounded asynchronous HTTP entry point for the synchronous RAG service."""

import asyncio
from threading import Lock
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

from .agent import PolicyAgent
from .config import Settings
from .providers import safe_error
from .service import RAGService


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    library: str = "policy"
    scope: str = "all"
    province: str = ""
    previous_question: str = Field(default="", max_length=500)
    mode: str = "direct"


def create_app(settings=None, service=None):
    settings = settings or (service.settings if service is not None else Settings.from_env())
    app = FastAPI(title="双碳政策 RAG API", version="1.0")
    semaphore = asyncio.Semaphore(settings.api_max_concurrent)
    init_lock = Lock()
    state = {"service": service}

    def get_service():
        if state["service"] is None:
            with init_lock:
                if state["service"] is None:
                    state["service"] = RAGService(settings)
        return state["service"]

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/v1/ask")
    async def ask(body: AskRequest, response: Response):
        if body.mode not in {"direct", "agent"} or body.library not in {"policy", "reference"}:
            raise HTTPException(422, "mode 或 library 无效")
        if body.scope not in {"all", "national", "province"}:
            raise HTTPException(422, "scope 无效")
        request_id = uuid4().hex
        started = perf_counter()
        try:
            await asyncio.wait_for(semaphore.acquire(), timeout=settings.api_queue_timeout)
        except TimeoutError as exc:
            raise HTTPException(503, "服务繁忙，排队超时，请稍后重试") from exc
        try:
            svc = await asyncio.to_thread(get_service)
            worker = PolicyAgent(svc) if body.mode == "agent" else svc
            result = await asyncio.to_thread(
                worker.ask,
                body.question,
                body.library,
                body.scope,
                body.province,
                body.previous_question,
            )
            elapsed = perf_counter() - started
            svc.ledger.record(
                "api_request",
                result.status,
                elapsed,
                details={"request_id": request_id, "mode": body.mode},
            )
            response.headers["X-Request-ID"] = request_id
            if result.status == "error":
                response.status_code = 502
            return {
                "request_id": request_id,
                "status": result.status,
                "answer": result.text,
                "elapsed_seconds": round(elapsed, 4),
                "citations": result.citations,
                "steps": result.steps,
            }
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        except Exception as exc:
            raise HTTPException(500, safe_error(exc)) from exc
        finally:
            semaphore.release()

    return app


app = create_app()
