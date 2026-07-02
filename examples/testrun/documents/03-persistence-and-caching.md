# Persistence and caching

When persistence is enabled, the indexer uses Pathway's built-in persistence
backend together with a call cache on the embedding step. On restart, Pathway
replays its persisted state and re-emits only the differences since the last run.

This has two effects. First, unchanged documents are not re-embedded, because the
embedder cache and the operator state are restored. Second, documents that were
removed while the indexer was down are correctly retracted from the vector
database on the next run.

vetosh also keeps a separate, persistent parse cache. It stores extracted
document text keyed by a stable serialization of the file metadata, so a
re-discovered unchanged file is not re-parsed. Entries are evicted when a
document is removed, which keeps the cache bounded. Disabling persistence still
produces identical vectors for a given set of files; it only forgoes the
cross-restart diffing and caching.
