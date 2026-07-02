# Pathway live data framework

vetosh is built on the Pathway live data framework. Pathway uses differential
dataflow: data flows through the computation graph as diffs (+1 for an addition,
-1 for a removal), and changes propagate incrementally with minimal recomputation.

When a file changes, Pathway emits a removal of the old version and an addition
of the new one. vetosh reads sources with the `only_metadata` mode, so the graph
holds only file paths and metadata, never the file bytes. A one gigabyte corpus
therefore stays just a few kilobytes inside the pipeline; the bytes are fetched
and parsed on demand.

Deletions are handled natively. The parse step is non-deterministic, so Pathway
memoizes its output and re-emits it negated when a file is removed, without ever
re-reading the now-deleted file. The retraction flows through chunking and
embedding, and the vector database sink removes exactly the matching vectors.
