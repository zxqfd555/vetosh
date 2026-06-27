"""Abstract base class for async vector-DB retrieval."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class AsyncVectorAccessor(ABC):
    """Retrieve the nearest chunks for a query embedding from a vector store.

    Implementations are async so the network-bound call to the vector DB does
    not block the FastAPI event loop.
    """

    @abstractmethod
    async def retrieve(self, embedding: list[float], k: int) -> list[dict[str, Any]]:
        """Return up to ``k`` results ordered by descending similarity.

        Each result is a dict ``{"text": str, "metadata": dict, "score": float}``
        where ``score`` is a cosine similarity in ``[-1, 1]`` (higher is closer).
        """

    @abstractmethod
    async def close(self) -> None:
        """Release any underlying connections / pools."""

    async def __aenter__(self) -> "AsyncVectorAccessor":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()
