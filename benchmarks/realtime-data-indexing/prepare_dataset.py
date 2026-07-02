#!/usr/bin/env python3
"""Prepare a plain-text benchmark corpus of a given size.

Downloads Wikipedia parquet shards from Hugging Face (public, no auth, fast
CDN) and explodes articles into one ``.txt`` file each until the requested
size is reached — so the same script scales from 100 MB to 10 GB by changing
``--size-mb``. Also emits ``questions.json``: distinctive articles sampled
across the corpus with queries that must retrieve exactly that file
(retrieval-accuracy check for the benchmark).

Usage:
    python prepare_dataset.py --size-mb 100
    # -> datasets/100mb/docs/*.txt + datasets/100mb/questions.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

HF_REPO = "wikimedia/wikipedia"
HF_CONFIG = "20231101.en"
# Sample a question article roughly every N articles, up to a cap.
QUESTION_EVERY = 2000
MAX_QUESTIONS = 20
MIN_ARTICLE_CHARS = 2000


def _slug(title: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", title).strip("-")[:80] or "article"


def _first_sentence(text: str) -> str:
    head = text[:500].replace("\n", " ")
    match = re.search(r"[.!?]\s", head)
    return head[: match.end()].strip() if match else head[:200].strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size-mb", type=int, default=100)
    parser.add_argument("--out", default=None, help="Output dir (default datasets/<size>mb)")
    args = parser.parse_args()

    from huggingface_hub import HfApi, hf_hub_download

    root = Path(__file__).parent
    out = Path(args.out) if args.out else root / "datasets" / f"{args.size_mb}mb"
    docs = out / "docs"
    docs.mkdir(parents=True, exist_ok=True)

    target_bytes = args.size_mb * 1024 * 1024
    shards = sorted(
        f
        for f in HfApi().list_repo_files(HF_REPO, repo_type="dataset")
        if f.startswith(f"{HF_CONFIG}/train-")
    )

    import pyarrow.parquet as pq

    written = 0
    article_index = 0
    questions: list[dict] = []
    for shard in shards:
        if written >= target_bytes:
            break
        print(f"downloading {shard} ...", flush=True)
        local = hf_hub_download(HF_REPO, shard, repo_type="dataset")
        table = pq.read_table(local, columns=["title", "text"])
        for title, text in zip(
            table.column("title").to_pylist(), table.column("text").to_pylist()
        ):
            if written >= target_bytes:
                break
            if not text or len(text) < 200:
                continue
            name = f"{article_index:07d}-{_slug(title)}.txt"
            payload = f"{title}\n\n{text}"
            (docs / name).write_text(payload, encoding="utf-8")
            written += len(payload.encode("utf-8"))
            if (
                article_index % QUESTION_EVERY == 0
                and len(questions) < MAX_QUESTIONS
                and len(text) >= MIN_ARTICLE_CHARS
            ):
                questions.append(
                    {
                        "query": f"{title}: {_first_sentence(text)}",
                        "expected_file": name,
                    }
                )
            article_index += 1

    (out / "questions.json").write_text(json.dumps(questions, indent=2))
    print(
        f"done: {article_index} files, {written / 1024 / 1024:.1f} MB, "
        f"{len(questions)} accuracy questions -> {out}"
    )


if __name__ == "__main__":
    main()
