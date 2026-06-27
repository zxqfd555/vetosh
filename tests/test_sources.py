"""Tests for multi-source support (filesystem + Google Drive, only_metadata)."""

from __future__ import annotations

from vetosh.config.schema import FsSource, GDriveSource, VetoshConfig
from vetosh.indexer.sources import FsFetcher, GDriveFetcher, make_fetcher


# -- schema ------------------------------------------------------------------


def test_gdrive_source_validates():
    src = GDriveSource(
        object_id="abc123",
        service_user_credentials_file="./creds.json",
        file_name_pattern="*.pdf",
    )
    assert src.type == "gdrive"
    assert src.mode == "streaming"


def test_mixed_sources_discriminated():
    config = VetoshConfig.model_validate(
        {
            "sources": [
                {"type": "fs", "path": "/data/docs", "glob": "**/*"},
                {
                    "type": "gdrive",
                    "object_id": "folder-id",
                    "service_user_credentials_file": "./creds.json",
                },
            ],
            "vector_db": {"type": "sqlite", "path": "/tmp/x.db"},
            "embedder": {"type": "mock"},
        }
    )
    config.for_indexer()
    assert isinstance(config.sources[0], FsSource)
    assert isinstance(config.sources[1], GDriveSource)


# -- fetchers ----------------------------------------------------------------


def test_fs_fetcher_reads_bytes_and_suffix(tmp_path):
    path = tmp_path / "doc.txt"
    path.write_bytes(b"hello world")
    contents, suffix = FsFetcher().fetch({"path": str(path)})
    assert contents == b"hello world"
    assert suffix == ".txt"


def test_gdrive_fetcher_uses_injected_download():
    calls = []

    def fake_download(file_id: str) -> bytes:
        calls.append(file_id)
        return b"%PDF-fake-bytes"

    fetcher = GDriveFetcher("./creds.json", download_fn=fake_download)
    contents, suffix = fetcher.fetch({"id": "file-42", "name": "report.pdf"})
    assert contents == b"%PDF-fake-bytes"
    assert suffix == ".pdf"
    assert calls == ["file-42"]


def test_make_fetcher_dispatch():
    assert isinstance(make_fetcher(FsSource(path="/x")), FsFetcher)
    gd = make_fetcher(
        GDriveSource(object_id="x", service_user_credentials_file="./c.json")
    )
    assert isinstance(gd, GDriveFetcher)
