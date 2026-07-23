"""Backend-agnostic unit tests for the shared in-process BM25 hybrid mixin
(:class:`serviette.server.hybrid.KeywordHybridMixin`), driven by an in-memory
fake accessor so the fusion / rebuild / size-cap logic is verified without any
real vector store."""

from __future__ import annotations

import asyncio
import logging

from serviette.server.accessors.abstract import AsyncVectorAccessor
from serviette.server.hybrid import KeywordHybridMixin


class _Config:
    def __init__(self, hybrid=False, hybrid_max_chunks=5_000_000):
        self.hybrid = hybrid
        self.hybrid_max_chunks = hybrid_max_chunks


class FakeHybridAccessor(KeywordHybridMixin, AsyncVectorAccessor):
    """A fake store: the "vector" query returns rows in insertion order; the
    hybrid hooks expose the same rows for the BM25 leg."""

    def __init__(self, rows, config):
        # rows: list of (text, embedding) — metadata omitted for brevity.
        self._rows = rows
        self.fetch_all_calls = 0
        self._init_hybrid(config)

    async def retrieve(self, embedding, k):
        return await self.retrieve_ex(embedding, k)

    async def retrieve_ex(self, embedding, k, *, query_text=None, with_embeddings=False):
        # "Vector" order = insertion order, descending fake score.
        vector_hits = []
        for i, (text, emb) in enumerate(self._rows[:k]):
            hit = {"text": text, "metadata": {}, "score": 1.0 - i * 0.01}
            if with_embeddings:
                hit["embedding"] = emb
            vector_hits.append(hit)
        return await self._fuse(vector_hits, query_text, k, with_embeddings)

    async def _hybrid_count(self):
        return len(self._rows)

    async def _hybrid_fetch_all(self, with_embeddings):
        self.fetch_all_calls += 1
        hits = []
        for text, emb in self._rows:
            hit = {"text": text, "metadata": {}}
            if with_embeddings:
                hit["embedding"] = emb
            hits.append(hit)
        return hits

    async def close(self):
        return None


ROWS = [
    ("alpha cats streaming engine", [1.0, 0.0]),
    ("beta dogs framework", [0.0, 1.0]),
    ("gamma Guadalupe Hidalgo 1848 treaty", [0.5, 0.5]),
]


def test_hybrid_off_is_pure_vector():
    acc = FakeHybridAccessor(ROWS, _Config(hybrid=False))
    hits = asyncio.run(acc.retrieve_ex([0.0, 0.0], 2, query_text="Guadalupe Hidalgo"))
    # Pure vector order (insertion), keyword leg never consulted.
    assert [h["text"] for h in hits] == [ROWS[0][0], ROWS[1][0]]
    assert acc.fetch_all_calls == 0


def test_hybrid_surfaces_keyword_match_vector_missed():
    acc = FakeHybridAccessor(ROWS, _Config(hybrid=True))
    # Vector top-2 would be rows 0,1; the keyword leg pulls in the Guadalupe
    # Hidalgo chunk (row 2), which RRF then fuses into the top-2.
    hits = asyncio.run(
        acc.retrieve_ex([0.0, 0.0], 2, query_text="when was Guadalupe Hidalgo signed")
    )
    assert any("Guadalupe Hidalgo" in h["text"] for h in hits)


def test_hybrid_no_query_text_stays_vector():
    acc = FakeHybridAccessor(ROWS, _Config(hybrid=True))
    hits = asyncio.run(acc.retrieve_ex([0.0, 0.0], 2, query_text=None))
    assert [h["text"] for h in hits] == [ROWS[0][0], ROWS[1][0]]
    assert acc.fetch_all_calls == 0


def test_index_rebuilds_only_when_count_changes():
    acc = FakeHybridAccessor(list(ROWS), _Config(hybrid=True))

    async def run():
        await acc.retrieve_ex([0.0, 0.0], 2, query_text="cats")
        await acc.retrieve_ex([0.0, 0.0], 2, query_text="dogs")
        calls_before = acc.fetch_all_calls
        acc._rows.append(("delta new zanzibar spice", [0.2, 0.2]))
        await acc.retrieve_ex([0.0, 0.0], 2, query_text="zanzibar")
        return calls_before

    calls_before = asyncio.run(run())
    # Built once for the first two queries (same count), rebuilt after growth.
    assert calls_before == 1
    assert acc.fetch_all_calls == 2


def test_size_cap_skips_keyword_leg(caplog):
    acc = FakeHybridAccessor(ROWS, _Config(hybrid=True, hybrid_max_chunks=1))
    with caplog.at_level(logging.WARNING):
        hits = asyncio.run(
            acc.retrieve_ex([0.0, 0.0], 2, query_text="Guadalupe Hidalgo")
        )
    # Above the cap: pure vector, one-time warning, no corpus fetch.
    assert [h["text"] for h in hits] == [ROWS[0][0], ROWS[1][0]]
    assert acc.fetch_all_calls == 0
    assert any("hybrid_max_chunks" in r.message for r in caplog.records)


def test_mmr_path_carries_embeddings_through_fusion():
    acc = FakeHybridAccessor(ROWS, _Config(hybrid=True))
    hits = asyncio.run(
        acc.retrieve_ex([0.0, 0.0], 3, query_text="cats", with_embeddings=True)
    )
    # Every fused hit keeps its embedding (needed downstream by MMR).
    assert all("embedding" in h for h in hits)
