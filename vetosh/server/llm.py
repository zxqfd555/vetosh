"""Optional LLM chat for the ``/rag`` endpoint.

Like the server embedder, this calls the provider SDK directly as a coroutine
rather than going through a Pathway LLM-chat UDF: ``/rag`` here is pure
retrieval + a single chat completion and requires no Pathway graph, connectors
or UDFs, which is the condition under which the spec allows shipping it in v1.

If you instead need streaming, multi-step agents, rerankers or anything that
benefits from Pathway's incremental graph, that belongs in a separate Pathway
service — left as a TODO and intentionally out of this decoupled server.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

_DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's question using only the "
    "provided context. If the context is insufficient, say so."
)


@runtime_checkable
class AsyncLLM(Protocol):
    async def complete(self, query: str, context: list[str]) -> str:
        ...

    async def close(self) -> None:
        ...


def _build_prompt(query: str, context: list[str]) -> str:
    joined = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(context))
    return f"Context:\n{joined}\n\nQuestion: {query}"


class MockLLM:
    """Offline LLM scaffold: echoes the retrieved context (no provider needed).

    Lets the full RAG/chat flow run with no API key — useful for testing the
    indexer → server → frontend stack end-to-end. Replace with a real ``llm``
    (e.g. ``openai``) for actual generated answers.
    """

    async def complete(self, query: str, context: list[str]) -> str:
        if not context:
            return (
                "(mock LLM) No relevant context was retrieved for your question. "
                "This is the offline scaffold — set an `llm` of type `openai` for "
                "real answers."
            )
        snippet = " ".join(context[0].split())
        if len(snippet) > 320:
            snippet = snippet[:320] + "…"
        return (
            f"(mock LLM) Based on {len(context)} retrieved snippet(s), the most "
            f'relevant one says:\n\n"{snippet}"\n\n'
            "Configure an `llm` of type `openai` for a real generated answer."
        )

    async def close(self) -> None:
        return None


class OpenAIChat:
    def __init__(self, config) -> None:
        self._model = config.model or "gpt-4o-mini"
        self._api_key = config.api_key
        extra = config.model_dump(exclude={"type", "model", "api_key"})
        self._client_kwargs = {k: v for k, v in extra.items() if v is not None}
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(api_key=self._api_key, **self._client_kwargs)
        return self._client

    async def complete(self, query: str, context: list[str]) -> str:
        client = self._ensure_client()
        resp = await client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": _DEFAULT_SYSTEM_PROMPT},
                {"role": "user", "content": _build_prompt(query, context)},
            ],
        )
        return resp.choices[0].message.content or ""

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None


_OPENAI_COMPATIBLE = {"openai", "litellm"}


def build_llm(config) -> AsyncLLM:
    if config.type == "mock":
        return MockLLM()
    if config.type in _OPENAI_COMPATIBLE:
        return OpenAIChat(config)
    raise ValueError(
        f"Server-side LLM for type {config.type!r} is not implemented. "
        "Supported: " + ", ".join(sorted(_OPENAI_COMPATIBLE))
    )
