"""Entrypoint for ``vetosh indexer``."""

from __future__ import annotations

import argparse
import logging

from vetosh.indexer.config import load_indexer_config
from vetosh.indexer.graph import run_indexer


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="vetosh indexer")
    parser.add_argument("--config", required=True, help="Path to the YAML config file")
    parser.add_argument(
        "--log-level", default="INFO", help="Logging level (default: INFO)"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=args.log_level.upper())
    config = load_indexer_config(args.config)
    run_indexer(config)


if __name__ == "__main__":
    main()
