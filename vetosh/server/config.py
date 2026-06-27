"""Config loading/validation for the server command."""

from __future__ import annotations

from pathlib import Path

from vetosh.config.schema import VetoshConfig, load_config


def load_server_config(path: str | Path) -> VetoshConfig:
    """Load a config file and assert it has what the server needs."""

    return load_config(path).for_server()
