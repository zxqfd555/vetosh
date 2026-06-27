"""Tests for the FastAPI server (uses the SQLite accessor + mock embedder)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from tests.conftest import fake_embedding, write_sqlite_rows
from vetosh.config.schema import EmbedderConfig, SqliteConfig, VetoshConfig
from vetosh.server.accessors.sqlite import SqliteAccessor
from vetosh.server.main import create_app

DOCS = ["alpha document about cats", "beta report on dogs", "gamma notes on birds"]


@pytest.fixture
def store_path(tmp_path):
    path = tmp_path / "store.sqlite3"
    rows = [
        {
            "id": str(i),
            "text": text,
            "metadata": {"path": f"/docs/{i}.txt", "idx": i},
            "embedding": fake_embedding(text),
        }
        for i, text in enumerate(DOCS)
    ]
    write_sqlite_rows(path, rows)
    return path


def _client(store_path, embedder, llm=None):
    config = VetoshConfig(
        vector_db=SqliteConfig(type="sqlite", path=str(store_path)),
        embedder=EmbedderConfig(type="openai"),
    )
    accessor = SqliteAccessor(config.vector_db)
    app = create_app(config, embedder=embedder, accessor=accessor, llm=llm)
    return TestClient(app)


def test_retrieve_top_k(store_path, mock_server_embedder):
    with _client(store_path, mock_server_embedder) as client:
        resp = client.post("/retrieve", json={"query": DOCS[0], "k": 2})
    assert resp.status_code == 200
    results = resp.json()["results"]
    assert len(results) == 2
    # Exact-text query embeds to the same vector -> top hit is that document.
    assert results[0]["text"] == DOCS[0]
    assert results[0]["score"] == pytest.approx(1.0, abs=1e-6)
    assert results[0]["metadata"]["idx"] == 0


def test_results_ordered_by_score_desc(store_path, mock_server_embedder):
    with _client(store_path, mock_server_embedder) as client:
        resp = client.post("/retrieve", json={"query": DOCS[1], "k": 3})
    scores = [r["score"] for r in resp.json()["results"]]
    assert scores == sorted(scores, reverse=True)


def test_rag_returns_answer(store_path, mock_server_embedder, mock_llm):
    with _client(store_path, mock_server_embedder, llm=mock_llm) as client:
        resp = client.post("/rag", json={"query": "tell me about cats", "k": 2})
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body["answer"], str) and body["answer"]
    assert len(body["sources"]) == 2


def test_rag_disabled_without_llm(store_path, mock_server_embedder):
    with _client(store_path, mock_server_embedder) as client:
        resp = client.post("/rag", json={"query": "x", "k": 1})
    assert resp.status_code == 501
