"""FastAPI web chat frontend.

A thin, decoupled service: it serves the single-page chat UI and *proxies*
requests to the configured vetosh API (``frontend.api_url``). The browser only
ever talks to this frontend's own origin, so there is no CORS to configure and
the API address stays server-side. Indexing, API and frontend can therefore each
run on a separate worker/host.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

from vetosh.config.schema import VetoshConfig

_STATIC_DIR = Path(__file__).parent / "static"
_INDEX_FILE = _STATIC_DIR / "index.html"

# Endpoints on the upstream API that the frontend is allowed to proxy.
_PROXY_ROUTES = {"rag": "/rag", "retrieve": "/retrieve"}


def _load_index(title: str) -> str:
    return _INDEX_FILE.read_text(encoding="utf-8").replace("__APP_TITLE__", title)


def create_app(config: VetoshConfig, *, client: httpx.AsyncClient | None = None) -> FastAPI:
    """Build the frontend app.

    ``client`` can be injected (tests provide an httpx client backed by a mock
    transport); otherwise one is created against ``frontend.api_url``.
    """

    config.for_frontend()
    fe = config.frontend
    index_html = _load_index(fe.title)
    owns_client = client is None

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        nonlocal client
        if client is None:
            client = httpx.AsyncClient(base_url=fe.api_url, timeout=120.0)
        try:
            yield
        finally:
            if owns_client and client is not None:
                await client.aclose()

    app = FastAPI(title=f"{fe.title} frontend", lifespan=lifespan)

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        return HTMLResponse(index_html)

    @app.get("/api/config")
    async def frontend_config() -> JSONResponse:
        # Informational only (shown in the header); no secrets here.
        return JSONResponse({"title": fe.title, "api_url": fe.api_url})

    async def _proxy(name: str, request: Request) -> JSONResponse:
        try:
            body = await request.json()
        except Exception:
            body = {}
        try:
            upstream = await client.post(_PROXY_ROUTES[name], json=body)
        except httpx.HTTPError as exc:
            return JSONResponse(
                {"detail": f"Could not reach the API at {fe.api_url}: {exc}"},
                status_code=502,
            )
        try:
            payload = upstream.json()
        except ValueError:
            payload = {"detail": upstream.text}
        return JSONResponse(payload, status_code=upstream.status_code)

    @app.post("/api/rag")
    async def rag(request: Request) -> JSONResponse:
        return await _proxy("rag", request)

    @app.post("/api/retrieve")
    async def retrieve(request: Request) -> JSONResponse:
        return await _proxy("retrieve", request)

    return app


def run(config: VetoshConfig) -> None:
    """Start uvicorn with the configured frontend host/port (used by the CLI)."""

    import uvicorn

    config.for_frontend()
    app = create_app(config)
    uvicorn.run(app, host=config.frontend.host, port=config.frontend.port)
