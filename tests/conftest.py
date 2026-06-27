"""Shared test fixtures.

Kept import-light at module scope: ``pathway`` is imported lazily inside the
fixtures that need it, so the cache/server/quickstart tests run without pulling
in the heavy Pathway/xpack stack.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

# Single source of truth for the deterministic fake embedding, shared with the
# indexer's test-only "mock" embedder so indexer and server vectors agree.
from vetosh.testing import EMBED_DIM, fake_embedding  # noqa: F401


class MockServerEmbedder:
    """Mock implementing the server's :class:`AsyncEmbedder` protocol."""

    async def embed(self, text: str) -> list[float]:
        return fake_embedding(text)

    async def close(self) -> None:
        return None


class MockLLM:
    """Mock implementing the server's :class:`AsyncLLM` protocol."""

    async def complete(self, query: str, context: list[str]) -> str:
        return f"Answer to {query!r} from {len(context)} source(s)."

    async def close(self) -> None:
        return None


@pytest.fixture
def mock_server_embedder() -> MockServerEmbedder:
    return MockServerEmbedder()


@pytest.fixture
def mock_llm() -> MockLLM:
    return MockLLM()


@pytest.fixture
def mock_xpack_embedder():
    """A Pathway UDF mock embedder for the indexer graph (returns ndarray)."""

    from vetosh.testing import build_mock_embedder

    return build_mock_embedder()


def write_sqlite_rows(path: Path, rows: list[dict]) -> None:
    """Populate a SQLite vector store directly (server tests)."""

    from vetosh.server.accessors.sqlite import connect

    conn = connect(path)
    for row in rows:
        conn.execute(
            "INSERT OR REPLACE INTO vetosh_embeddings (id, text, metadata, embedding) "
            "VALUES (?, ?, ?, ?)",
            (
                row["id"],
                row["text"],
                json.dumps(row.get("metadata", {})),
                json.dumps(row["embedding"]),
            ),
        )
    conn.commit()
    conn.close()
