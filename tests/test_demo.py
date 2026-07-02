"""Unit tests for the ``vetosh demo`` scaffolding (no processes spawned)."""

from __future__ import annotations

from pathlib import Path

from vetosh.config.schema import load_config_dict
from vetosh.demo import build_demo_config, choose_embedder, prepare_demo_dir


def test_demo_config_validates(tmp_path):
    config = build_demo_config(
        tmp_path, port=8123, embedder="mock", license_key="KEY"
    )
    cfg = load_config_dict(config)
    cfg.for_indexer()
    cfg.for_server()
    assert cfg.vector_db.type == "duckdb"
    assert cfg.server.port == 8123
    assert cfg.llm is not None  # chat always answers something


def test_demo_openai_choice_uses_env_refs(tmp_path):
    config = build_demo_config(
        tmp_path, port=8000, embedder="openai", license_key="${PATHWAY_LICENSE_KEY}"
    )
    assert config["embedder"]["api_key"] == "${OPENAI_API_KEY}"
    assert config["llm"]["type"] == "openai"


def test_choose_embedder_prefers_openai_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert choose_embedder() == "openai"


def test_prepare_demo_dir_copies_corpus_and_writes_config(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config_path = prepare_demo_dir(tmp_path, port=8000, license_key="KEY")
    docs = list((tmp_path / "docs").glob("*.md"))
    assert len(docs) >= 5
    assert any(p.name == "pricing.md" for p in docs)
    assert config_path.exists()
    # The generated file round-trips through the normal config loader.
    from vetosh.config.schema import load_config

    cfg = load_config(config_path)
    assert Path(cfg.sources[0].path) == tmp_path / "docs"
    # Idempotent: a second run must not fail on the existing docs dir.
    prepare_demo_dir(tmp_path, port=8000, license_key="KEY")
