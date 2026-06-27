"""Config loading/validation for the indexer command."""

from __future__ import annotations

from pathlib import Path

from vetosh.config.schema import VetoshConfig, load_config


def load_indexer_config(path: str | Path) -> VetoshConfig:
    """Load a config file and assert it has what the indexer needs."""

    return load_config(path).for_indexer()
