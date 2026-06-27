"""Shared configuration schema for vetosh (Pydantic v2).

A single YAML file can hold both indexer and server configuration (a *universal*
config), or the user can split it into an indexer-only and a server-only file.
Most top-level sections are therefore optional at the schema level; each
component validates the subset it needs via :func:`require` at startup.

String values support ``${ENV_VAR}`` interpolation, resolved while the YAML is
loaded (see :func:`load_config`). This keeps credentials out of the config file
itself, satisfying the "no hardcoded credentials" principle.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Annotated, Any, Literal, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from vetosh import APP_NAME

# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------


class FsSource(BaseModel):
    """A filesystem source watched by the indexer.

    ``type`` is the discriminator across source types. Only sources with an
    ``only_metadata`` read mode are supported (see ``GDriveSource``).
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["fs"] = "fs"
    path: str
    glob: str = "**/*"
    # "streaming" watches the path continuously (default); "static" reads the
    # current contents once and lets the graph terminate (used for one-shot
    # indexing and tests).
    mode: Literal["streaming", "static"] = "streaming"


class GDriveSource(BaseModel):
    """A Google Drive source (folder or file) read in ``only_metadata`` mode.

    Like the filesystem source, the graph holds only metadata; file bytes are
    downloaded on demand by id during parsing. Requires a Google service-account
    credentials file (see the Google Drive connector docs).
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["gdrive"] = "gdrive"
    object_id: str
    service_user_credentials_file: str
    # Glob (or list of globs) on file names, e.g. "*.pdf". None = no filtering.
    file_name_pattern: str | list[str] | None = None
    object_size_limit: int | None = None
    mode: Literal["streaming", "static"] = "streaming"


Source = Annotated[Union[FsSource, GDriveSource], Field(discriminator="type")]


# ---------------------------------------------------------------------------
# Vector databases
# ---------------------------------------------------------------------------


class PgVectorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["pgvector"] = "pgvector"
    connection_string: str
    table: str = "vetosh_embeddings"
    # Dimension of the embedding column. Used when the indexer auto-creates the
    # table. If omitted, it is inferred from the embedder at indexer startup.
    embedding_dimension: int | None = None


class MilvusConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["milvus"] = "milvus"
    # Either a full URI (e.g. "http://localhost:19530" or "./milvus.db" for
    # Milvus Lite) or host + port. ``uri`` takes precedence when provided.
    uri: str | None = None
    host: str = "localhost"
    port: int = 19530
    collection: str = "vetosh_embeddings"
    token: str | None = None
    embedding_dimension: int | None = None

    def resolved_uri(self) -> str:
        if self.uri:
            return self.uri
        return f"http://{self.host}:{self.port}"


class SqliteConfig(BaseModel):
    """Test-only vector store backed by a local SQLite file.

    Not documented for end users — it exists so the full indexer→server flow can
    be exercised without any external service.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["sqlite"] = "sqlite"
    path: str
    table: str = "vetosh_embeddings"


VectorDBConfig = Annotated[
    Union[PgVectorConfig, MilvusConfig, SqliteConfig],
    Field(discriminator="type"),
]


# ---------------------------------------------------------------------------
# Embedder / splitter / LLM
# ---------------------------------------------------------------------------


class EmbedderConfig(BaseModel):
    """Embedder configuration.

    ``type`` selects an embedder family that maps onto a
    ``pathway.xpacks.llm.embedders`` class on the indexer side and onto an async
    client on the server side. Any extra keys are forwarded to the underlying
    embedder constructor / client, so provider-specific options need no schema
    change.
    """

    model_config = ConfigDict(extra="allow")

    type: str = "openai"
    model: str | None = None
    api_key: str | None = None


class SplitterConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: str = "token_count"
    chunk_size: int = 512
    chunk_overlap: int = 50


class LLMConfig(BaseModel):
    """Optional LLM configuration for the ``/rag`` endpoint."""

    model_config = ConfigDict(extra="allow")

    type: str = "openai"
    model: str | None = None
    api_key: str | None = None


# ---------------------------------------------------------------------------
# Persistence / server
# ---------------------------------------------------------------------------


class PersistenceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    backend: Literal["filesystem"] = "filesystem"
    path: str = "/var/vetosh/persistence"


class ServerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    host: str = "0.0.0.0"
    port: int = 8000


class FrontendConfig(BaseModel):
    """Web chat UI. Decoupled from the API: it only needs the API's address.

    The frontend serves the page and *proxies* requests to ``api_url`` server-side
    (no CORS, the API address never reaches the browser), so indexing, API and
    frontend can each run on a different worker/host.
    """

    model_config = ConfigDict(extra="forbid")

    host: str = "0.0.0.0"
    port: int = 3000
    # Base URL of the vetosh server API this frontend talks to.
    api_url: str = "http://localhost:8000"
    title: str = APP_NAME


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------


class VetoshConfig(BaseModel):
    """Top-level universal configuration.

    Sections are optional so the same model validates indexer-only and
    server-only files. Use :meth:`for_indexer` / :meth:`for_server` to assert the
    fields a given component requires.
    """

    model_config = ConfigDict(extra="forbid")

    pathway_license_key: str | None = None

    sources: list[Source] = Field(default_factory=list)
    vector_db: VectorDBConfig | None = None
    embedder: EmbedderConfig | None = None
    splitter: SplitterConfig = Field(default_factory=SplitterConfig)

    persistence: PersistenceConfig = Field(default_factory=PersistenceConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    llm: LLMConfig | None = None
    frontend: FrontendConfig | None = None

    # -- per-component requirements -------------------------------------------

    def require(self, *fields: str) -> None:
        missing = [f for f in fields if not getattr(self, f)]
        if missing:
            raise ValueError(
                "Missing required config section(s) for this command: "
                + ", ".join(missing)
            )

    def for_indexer(self) -> VetoshConfig:
        self.require("sources", "vector_db", "embedder")
        return self

    def for_server(self) -> VetoshConfig:
        self.require("vector_db", "embedder")
        return self

    def for_frontend(self) -> VetoshConfig:
        self.require("frontend")
        return self


# ---------------------------------------------------------------------------
# Loading + ${ENV_VAR} interpolation
# ---------------------------------------------------------------------------

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def interpolate_env(value: Any) -> Any:
    """Recursively replace ``${VAR}`` occurrences in strings with env values.

    A missing variable resolves to an empty string and a warning-worthy state
    that surfaces later as a validation/connection error, rather than silently
    embedding the literal ``${VAR}`` text.
    """

    if isinstance(value, str):
        return _ENV_PATTERN.sub(lambda m: os.environ.get(m.group(1), ""), value)
    if isinstance(value, dict):
        return {k: interpolate_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [interpolate_env(v) for v in value]
    return value


def load_config(path: str | Path) -> VetoshConfig:
    """Load, env-interpolate and validate a YAML config file."""

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config root must be a mapping, got {type(raw).__name__}")
    interpolated = interpolate_env(raw)
    try:
        return VetoshConfig.model_validate(interpolated)
    except ValidationError as exc:  # pragma: no cover - re-raised with context
        raise ValueError(f"Invalid configuration in {path}:\n{exc}") from exc


def load_config_dict(data: dict[str, Any]) -> VetoshConfig:
    """Validate an already-parsed config dict (used by the quickstart wizard)."""

    return VetoshConfig.model_validate(interpolate_env(data))
