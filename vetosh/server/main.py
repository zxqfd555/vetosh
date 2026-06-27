"""FastAPI retrieval/RAG server.

Async and coroutine-based so network-bound calls to the vector DB and embedding
provider don't exhaust a thread pool. Fully decoupled from the indexer: it only
reads from the vector DB, so it can be scaled horizontally and independently.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from vetosh.config.schema import VetoshConfig
from vetosh.server.accessors import AsyncVectorAccessor, build_accessor
from vetosh.server.embedder import AsyncEmbedder, build_embedder
from vetosh.server.llm import AsyncLLM, build_llm


class RetrieveRequest(BaseModel):
    query: str
    k: int = Field(default=5, ge=1)


class RetrieveResult(BaseModel):
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    score: float


class RetrieveResponse(BaseModel):
    results: list[RetrieveResult]


class RagResponse(BaseModel):
    answer: str
    sources: list[RetrieveResult]


def create_app(
    config: VetoshConfig,
    *,
    embedder: AsyncEmbedder | None = None,
    accessor: AsyncVectorAccessor | None = None,
    llm: AsyncLLM | None = None,
) -> FastAPI:
    """Build the FastAPI app.

    The ``embedder``/``accessor``/``llm`` overrides exist for testing (inject a
    mock embedder and a SQLite accessor); in production they are built from the
    config.
    """

    config.for_server()

    embedder = embedder or build_embedder(config.embedder)
    accessor = accessor or build_accessor(config.vector_db)
    # ``/rag`` is enabled only when an LLM is configured (or injected).
    if llm is None and config.llm is not None:
        llm = build_llm(config.llm)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            yield
        finally:
            await accessor.close()
            await embedder.close()
            if llm is not None:
                await llm.close()

    app = FastAPI(title="vetosh server", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/retrieve", response_model=RetrieveResponse)
    async def retrieve(req: RetrieveRequest) -> RetrieveResponse:
        embedding = await embedder.embed(req.query)
        hits = await accessor.retrieve(embedding, req.k)
        return RetrieveResponse(results=[RetrieveResult(**h) for h in hits])

    @app.post("/rag", response_model=RagResponse)
    async def rag(req: RetrieveRequest) -> RagResponse:
        if llm is None:
            raise HTTPException(
                status_code=501,
                detail="The /rag endpoint requires an 'llm' config section.",
            )
        embedding = await embedder.embed(req.query)
        hits = await accessor.retrieve(embedding, req.k)
        answer = await llm.complete(req.query, [h["text"] for h in hits])
        return RagResponse(
            answer=answer, sources=[RetrieveResult(**h) for h in hits]
        )

    return app


def run(config: VetoshConfig) -> None:
    """Start uvicorn with the configured host/port (used by the CLI)."""

    import uvicorn

    app = create_app(config)
    uvicorn.run(app, host=config.server.host, port=config.server.port)
