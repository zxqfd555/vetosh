"""Async MongoDB Atlas Vector Search accessor using ``pymongo``'s async API.

The indexer's ``pw.io.mongodb`` snapshot sink stores one document per chunk
(``chunk_id``, ``text``, ``metadata``, ``embedding`` as a number array).
Retrieval runs an Atlas ``$vectorSearch`` aggregation against the configured
``vectorSearch`` index (create it on the ``embedding`` path with
``numDimensions`` matching the embedder and the cosine similarity).

Atlas normalizes the cosine score to ``(1 + cosine) / 2`` ∈ [0, 1]; converted
back to a plain cosine similarity here to match the accessor contract.
"""

from __future__ import annotations

import json
from typing import Any

from serviette.server.accessors.abstract import AsyncVectorAccessor

# Atlas recommends ~10-20x the limit for the ANN candidate pool.
_CANDIDATE_FACTOR = 15


class MongoDbAccessor(AsyncVectorAccessor):
    def __init__(self, config) -> None:
        self._config = config
        self._client = None

    def _ensure_collection(self):
        if self._client is None:
            from pymongo import AsyncMongoClient

            self._client = AsyncMongoClient(self._config.connection_string)
        return self._client[self._config.database][self._config.collection]

    async def retrieve(self, embedding: list[float], k: int) -> list[dict[str, Any]]:
        collection = self._ensure_collection()
        pipeline = [
            {
                "$vectorSearch": {
                    "index": self._config.vector_index,
                    "path": "embedding",
                    "queryVector": list(map(float, embedding)),
                    "numCandidates": max(k * _CANDIDATE_FACTOR, 100),
                    "limit": k,
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "text": 1,
                    "metadata": 1,
                    "score": {"$meta": "vectorSearchScore"},
                }
            },
        ]
        results: list[dict[str, Any]] = []
        async for doc in await collection.aggregate(pipeline):
            # The sink stores the pw.Json metadata as a JSON string in BSON.
            metadata = doc.get("metadata")
            if isinstance(metadata, str):
                metadata = json.loads(metadata)
            # Undo Atlas's (1 + cosine) / 2 normalization.
            results.append(
                {
                    "text": doc.get("text", ""),
                    "metadata": metadata or {},
                    "score": 2.0 * float(doc.get("score", 0.5)) - 1.0,
                }
            )
        return results

    async def stats(self) -> dict[str, Any]:
        collection = self._ensure_collection()
        return {"chunks": int(await collection.estimated_document_count())}

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None
