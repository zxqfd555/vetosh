"""Tests for the persistent parse cache."""

from __future__ import annotations

from vetosh.indexer.cache import ParseCache, cache_key


def test_store_and_retrieve(tmp_path):
    cache = ParseCache(tmp_path)
    meta = {"path": "/docs/a.pdf", "size": 10, "modified_at": 111}
    cache.store(meta, "parsed text")
    assert cache.get(meta) == "parsed text"
    cache.close()


def test_delete_on_removal(tmp_path):
    cache = ParseCache(tmp_path)
    meta = {"path": "/docs/a.pdf", "size": 10}
    cache.store(meta, "parsed text")
    # delete returns the cached value so the sink can retract old vectors...
    assert cache.delete(meta) == "parsed text"
    # ...and the entry is gone afterwards (bounded growth).
    assert cache.get(meta) is None
    assert cache.delete(meta) is None
    cache.close()


def test_key_stable_regardless_of_order():
    a = {"path": "/x", "size": 1, "modified_at": 9}
    b = {"modified_at": 9, "size": 1, "path": "/x"}
    assert cache_key(a) == cache_key(b)


def test_key_distinguishes_metadata():
    a = {"path": "/x", "modified_at": 1}
    b = {"path": "/x", "modified_at": 2}
    assert cache_key(a) != cache_key(b)


def test_persistence_across_restart(tmp_path):
    meta = {"path": "/docs/a.pdf", "size": 42}
    cache = ParseCache(tmp_path)
    cache.store(meta, "remembered")
    cache.close()

    # Re-open the same directory in a fresh instance (simulating a restart).
    reopened = ParseCache(tmp_path)
    assert reopened.get(meta) == "remembered"
    reopened.close()


def test_env_var_cache_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("VETOSH_CACHE_DIR", str(tmp_path / "fromenv"))
    cache = ParseCache()
    assert cache.directory == tmp_path / "fromenv"
    cache.close()
