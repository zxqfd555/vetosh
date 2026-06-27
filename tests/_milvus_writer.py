"""Subprocess driver for the Milvus integration test.

Every operation that touches the Milvus Lite file runs in its own short-lived
process via this driver, so two processes never hold the embedded server's file
lock at once (Milvus Lite releases the lock only on process exit, not reliably on
``close()``). Results are printed to stdout as JSON.

Usage:
  python -m tests._milvus_writer write    <uri> <collection> <config-json>
  python -m tests._milvus_writer paths    <uri> <collection>
  python -m tests._milvus_writer retrieve <uri> <collection> <query>
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path


def _ensure_collection(uri: str, collection: str) -> None:
    from pymilvus import DataType, MilvusClient

    from vetosh.testing import EMBED_DIM

    client = MilvusClient(uri=uri)
    try:
        if client.has_collection(collection):
            return
        schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("chunk_id", DataType.VARCHAR, is_primary=True, max_length=128)
        schema.add_field("text", DataType.VARCHAR, max_length=65535)
        schema.add_field("metadata", DataType.JSON)
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=EMBED_DIM)
        index = client.prepare_index_params()
        index.add_index(field_name="embedding", index_type="FLAT", metric_type="COSINE")
        client.create_collection(collection, schema=schema, index_params=index)
    finally:
        client.close()


def _write(uri: str, collection: str, config_json: str) -> object:
    import pathway as pw

    from vetosh.config.schema import VetoshConfig
    from vetosh.indexer.graph import build_graph, persistence_config

    _ensure_collection(uri, collection)
    config = VetoshConfig.model_validate(json.loads(config_json))
    build_graph(config)
    pw.run(
        persistence_config=persistence_config(config),
        monitoring_level=pw.MonitoringLevel.NONE,
    )
    return {"ok": True}


def _paths(uri: str, collection: str) -> object:
    from pymilvus import MilvusClient

    client = MilvusClient(uri=uri)
    try:
        client.load_collection(collection)
        rows = client.query(collection, filter="", output_fields=["metadata"], limit=1000)
    finally:
        client.close()
    return sorted(Path(r["metadata"]["path"]).name for r in rows)


def _retrieve(uri: str, collection: str, query: str) -> object:
    from vetosh.config.schema import MilvusConfig
    from vetosh.server.accessors.milvus import MilvusAccessor
    from vetosh.testing import fake_embedding

    async def go():
        accessor = MilvusAccessor(MilvusConfig(uri=uri, collection=collection))
        try:
            return await accessor.retrieve(fake_embedding(query), k=5)
        finally:
            await accessor.close()

    return asyncio.run(go())


def main() -> None:
    action = sys.argv[1]
    if action == "write":
        result = _write(sys.argv[2], sys.argv[3], sys.argv[4])
    elif action == "paths":
        result = _paths(sys.argv[2], sys.argv[3])
    elif action == "retrieve":
        result = _retrieve(sys.argv[2], sys.argv[3], sys.argv[4])
    else:  # pragma: no cover
        raise SystemExit(f"unknown action {action!r}")
    print("__RESULT__" + json.dumps(result))


if __name__ == "__main__":
    main()
