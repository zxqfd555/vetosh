"""Query embedding for the server.

Design note — why not call the xpack embedder directly?
-------------------------------------------------------
``pathway.xpacks.llm.embedders`` classes are Pathway *UDFs*: calling them builds
graph nodes, not plain values, and they're meant to run inside ``pw.run``. The
server is a network-bound async FastAPS service and is deliberately decoupled
from Pathway (it need not even have ``pathway`` installed). So we embed a single
query by calling the provider SDK directly as a coroutine — exactly the
"coroutines avoid thread-pool exhaustion" rationale for choosing FastAPI.

The ``type``/``model`` config keys mirror the indexer's embedder config, so the
**same** embedder section produces matching vectors on both sides as long as the
model matches. Tests inject a deterministic mock implementing
:class:`AsyncEmbedder`.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class AsyncEmbedder(Protocol):
    async def embed(self, text: str) -> list[float]:
        ...

    async def close(self) -> None:
        ...


class OpenAIAsyncEmbedder:
    """Embed queries with OpenAI (or any OpenAI-compatible endpoint)."""

    def __init__(self, config) -> None:
        self._model = config.model or "text-embedding-3-small"
        self._api_key = config.api_key
        # Forward any extra keys (e.g. base_url) declared on the config.
        extra = config.model_dump(exclude={"type", "model", "api_key"})
        self._client_kwargs = {k: v for k, v in extra.items() if v is not None}
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(api_key=self._api_key, **self._client_kwargs)
        return self._client

    async def embed(self, text: str) -> list[float]:
        client = self._ensure_client()
        resp = await client.embeddings.create(model=self._model, input=[text])
        return list(resp.data[0].embedding)

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None


# Embedder families that map onto an OpenAI-compatible async client. Additional
# providers (gemini, bedrock, ...) can be added as dedicated classes here without
# touching the rest of the server.
_OPENAI_COMPATIBLE = {"openai", "litellm"}


def build_embedder(config) -> AsyncEmbedder:
    """Construct an :class:`AsyncEmbedder` from an ``EmbedderConfig``."""

    if config.type in _OPENAI_COMPATIBLE:
        return OpenAIAsyncEmbedder(config)
    raise ValueError(
        f"Server-side embedding for type {config.type!r} is not implemented. "
        "Supported on the server: " + ", ".join(sorted(_OPENAI_COMPATIBLE))
    )
