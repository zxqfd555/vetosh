"""Async vector-database accessors.

Each accessor implements :class:`AsyncVectorAccessor` and knows how to fetch the
top-k nearest chunks for a query embedding from one backend. The server is
decoupled from the indexer and talks to the vector DB only through these.
"""

from vetosh.server.accessors.abstract import AsyncVectorAccessor

__all__ = ["AsyncVectorAccessor", "build_accessor"]


def build_accessor(vector_db) -> AsyncVectorAccessor:
    """Construct the accessor matching a ``vector_db`` config section.

    Imports are local so installing only the backend you use (e.g. no
    ``pymilvus``) does not break the others.
    """

    db_type = vector_db.type
    if db_type == "pgvector":
        from vetosh.server.accessors.pgvector import PgVectorAccessor

        return PgVectorAccessor(vector_db)
    if db_type == "milvus":
        from vetosh.server.accessors.milvus import MilvusAccessor

        return MilvusAccessor(vector_db)
    if db_type == "sqlite":
        from vetosh.server.accessors.sqlite import SqliteAccessor

        return SqliteAccessor(vector_db)
    raise ValueError(f"Unsupported vector_db type: {db_type!r}")
