"""Tests for the quickstart wizard config generation."""

from __future__ import annotations

import yaml

from vetosh.config.schema import VetoshConfig, load_config_dict
from vetosh.quickstart.wizard import ScriptedPrompter, Wizard, build_config, dump_yaml

BASE = {
    "license_key": "test-license-key-123",
    "sources": [{"type": "fs", "path": "/data/docs", "glob": "**/*"}],
    "vector_db_type": "pgvector",
    "pg_connection_string": "postgresql://u:p@localhost/db",
    "collection": "vetosh_embeddings",
    "embedder_type": "openai",
    "embedder_model": "text-embedding-3-small",
    "embedder_api_key": "${OPENAI_API_KEY}",
    "chunk_size": 512,
    "chunk_overlap": 50,
    "persistence_enabled": True,
    "persistence_path": "/var/vetosh/persistence",
    "server_host": "0.0.0.0",
    "server_port": 8000,
    "rag_enabled": False,
    "output_path": "./config.yaml",
}


def _validate(answers: dict) -> VetoshConfig:
    config_dict = build_config(answers)
    # Round-trips through YAML the way the wizard writes it.
    reparsed = yaml.safe_load(dump_yaml(config_dict))
    return load_config_dict(reparsed)


def test_universal_config_valid():
    answers = {**BASE, "config_type": "universal"}
    cfg = _validate(answers)
    cfg.for_indexer()
    cfg.for_server()
    assert cfg.sources and cfg.server is not None


def test_indexer_only_config_valid():
    answers = {**BASE, "config_type": "indexer"}
    config_dict = build_config(answers)
    cfg = _validate(answers)
    cfg.for_indexer()
    assert "server" not in config_dict
    assert "llm" not in config_dict


def test_server_only_config_valid():
    answers = {**BASE, "config_type": "server"}
    config_dict = build_config(answers)
    cfg = _validate(answers)
    cfg.for_server()
    assert "sources" not in config_dict


def test_license_key_present_in_yaml():
    answers = {**BASE, "config_type": "universal"}
    text = dump_yaml(build_config(answers))
    assert "test-license-key-123" in text
    assert yaml.safe_load(text)["pathway_license_key"] == "test-license-key-123"


def test_rag_section_emitted_when_enabled():
    answers = {
        **BASE,
        "config_type": "server",
        "rag_enabled": True,
        "llm_type": "openai",
        "llm_model": "gpt-4o-mini",
        "llm_api_key": "${OPENAI_API_KEY}",
    }
    config_dict = build_config(answers)
    assert config_dict["llm"]["model"] == "gpt-4o-mini"


def test_milvus_vector_db():
    answers = {
        **BASE,
        "config_type": "indexer",
        "vector_db_type": "milvus",
        "milvus_uri": "http://localhost:19530",
    }
    cfg = _validate(answers)
    assert cfg.vector_db.type == "milvus"


def test_defaults_have_no_brand_name():
    """Suggested defaults must not embed the project name."""

    answers = {**BASE, "config_type": "universal"}
    # Drop explicit overrides so build_config falls back to its own defaults.
    for key in ("collection", "persistence_path"):
        answers.pop(key, None)
    config = build_config(answers)
    assert config["vector_db"]["table"] == "embeddings"
    assert "vetosh" not in config["persistence"]["path"]


def _feed(tokens):
    it = iter(tokens)
    return lambda _prompt: next(it)


def test_wizard_run_collects_multiple_sources():
    """Driving run() through the scripted prompter adds two sources via the loop."""

    tokens = [
        "2",                       # config type: indexer only
        "1", "/data/a", "",        # source 1: filesystem, path, default glob
        "1", "/data/b", "",        # source 2: filesystem, path, default glob
        "3",                       # add another? -> Done (fs, gdrive, done)
        "1",                       # vector db: pgvector
        "postgresql://u:p@h/db",   # pg connection string
        "",                        # collection (default)
        "1",                       # embedder: openai
        "", "",                    # embedder model, api key (defaults)
        "", "",                    # chunk size, overlap (defaults)
        "",                        # persistence enabled? Yes (default)
        "",                        # persistence path (default)
        "MY-KEY",                  # license key
        "",                        # output path (default)
    ]
    prompter = ScriptedPrompter(input_fn=_feed(tokens), output_fn=lambda _s: None)
    answers = Wizard(prompter=prompter).run()

    assert answers["config_type"] == "indexer"
    assert [s["path"] for s in answers["sources"]] == ["/data/a", "/data/b"]
    assert all(s["type"] == "fs" for s in answers["sources"])

    cfg = _validate(answers)
    cfg.for_indexer()
    assert len(cfg.sources) == 2


def test_wizard_run_collects_gdrive_source():
    """The sources loop can add a Google Drive source (second source type)."""

    tokens = [
        "1",                        # config type: universal
        "2",                        # source 1 type: Google Drive (index 2)
        "drive-folder-id",          # gdrive object id
        "./creds.json",             # credentials file
        "*.pdf",                    # file name pattern
        "3",                        # add another? -> Done (fs, gdrive, done) = index 3
        "1",                        # vector db: pgvector
        "postgresql://u:p@h/db",
        "",                         # collection default
        "1", "", "",                # embedder openai, model, key
        "", "",                     # chunk size/overlap
        "",                         # persistence yes
        "",                         # persistence path
        "0.0.0.0", "8000",          # server host/port
        "",                         # rag? No (default index 1 -> blank=default? confirm default False)
        "LIC",                      # license
        "",                         # output path
    ]
    prompter = ScriptedPrompter(input_fn=_feed(tokens), output_fn=lambda _s: None)
    answers = Wizard(prompter=prompter).run()

    assert len(answers["sources"]) == 1
    gd = answers["sources"][0]
    assert gd["type"] == "gdrive"
    assert gd["object_id"] == "drive-folder-id"
    assert gd["file_name_pattern"] == "*.pdf"

    cfg = _validate(answers)
    cfg.for_indexer()
