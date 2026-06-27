"""Source connectors and byte fetchers for the indexer.

We only support sources that have an ``only_metadata`` read mode (like the
filesystem connector): the Pathway graph holds just paths/identifiers and
metadata, never the file bytes, and the bytes are fetched on demand inside the
parse UDF. Today that is:

- **fs** — ``pw.io.fs.read(format="only_metadata")``; bytes read from the local
  path in the metadata.
- **gdrive** — ``pw.io.gdrive.read(format="only_metadata")``; bytes downloaded by
  file ``id`` via the Google Drive API.

S3/MinIO do not offer ``only_metadata`` and SharePoint has no connector in this
Pathway build, so they are intentionally out of scope here.

Each source type provides a :class:`Fetcher` that turns a metadata dict into
``(bytes, suffix)`` for the parser. The fetcher is what makes the parse step
source-agnostic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Protocol

import pathway as pw

# Drive scope sufficient to download file contents.
_GDRIVE_SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]


# ---------------------------------------------------------------------------
# Reading source tables in only_metadata mode
# ---------------------------------------------------------------------------


def read_source(src, name: str) -> pw.Table:
    """Return an ``only_metadata`` table for ``src`` (one ``_metadata`` column).

    ``name`` is the persistable connector name (unique per source).
    """

    if src.type == "fs":
        return pw.io.fs.read(
            src.path,
            format="only_metadata",
            mode=src.mode,
            object_pattern=src.glob,
            name=name,
        )
    if src.type == "gdrive":
        return pw.io.gdrive.read(
            src.object_id,
            format="only_metadata",
            mode=src.mode,
            service_user_credentials_file=src.service_user_credentials_file,
            file_name_pattern=src.file_name_pattern,
            name=name,
        )
    raise ValueError(f"Unsupported source type: {src.type!r}")  # pragma: no cover


# ---------------------------------------------------------------------------
# Byte fetchers
# ---------------------------------------------------------------------------


class Fetcher(Protocol):
    def fetch(self, metadata: dict[str, Any]) -> tuple[bytes, str]:
        """Return ``(contents, suffix)`` for the object described by ``metadata``."""


class FsFetcher:
    """Read bytes from the local filesystem path in the metadata."""

    def fetch(self, metadata: dict[str, Any]) -> tuple[bytes, str]:
        path = metadata["path"]
        return Path(path).read_bytes(), Path(path).suffix


class GDriveFetcher:
    """Download bytes from Google Drive by file id.

    ``download_fn`` is injectable for tests; in production it lazily builds a
    Drive client from the service-account credentials file.
    """

    def __init__(
        self,
        service_user_credentials_file: str,
        download_fn: Callable[[str], bytes] | None = None,
    ) -> None:
        self._credentials_file = service_user_credentials_file
        self._download_fn = download_fn
        self._service = None

    def _service_download(self, file_id: str) -> bytes:
        if self._service is None:
            from google.oauth2.service_account import Credentials
            from googleapiclient.discovery import build

            creds = Credentials.from_service_account_file(
                self._credentials_file, scopes=_GDRIVE_SCOPES
            )
            self._service = build("drive", "v3", credentials=creds)
        # get_media's execute() returns the raw file bytes.
        return self._service.files().get_media(fileId=file_id).execute()

    def fetch(self, metadata: dict[str, Any]) -> tuple[bytes, str]:
        file_id = metadata["id"]
        download = self._download_fn or self._service_download
        contents = download(file_id)
        # Derive the extension from the file name; Google-native docs (Docs/
        # Sheets) have no binary download and are skipped upstream.
        suffix = Path(metadata.get("name", "")).suffix
        return contents, suffix


def make_fetcher(src) -> Fetcher:
    if src.type == "fs":
        return FsFetcher()
    if src.type == "gdrive":
        return GDriveFetcher(src.service_user_credentials_file)
    raise ValueError(f"Unsupported source type: {src.type!r}")  # pragma: no cover
