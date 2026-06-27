"""Config loading/validation for the frontend command."""

from __future__ import annotations

from pathlib import Path

from vetosh.config.schema import VetoshConfig, load_config


def load_frontend_config(path: str | Path) -> VetoshConfig:
    """Load a config file and assert it has a ``frontend`` section."""

    return load_config(path).for_frontend()
