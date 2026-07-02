# Vector databases

vetosh can write embeddings into different vector databases, selected by the
`vector_db.type` field in the configuration.

- **pgvector** stores vectors in PostgreSQL using the pgvector extension. The
  indexer writes in snapshot mode keyed by a chunk id, so updates and deletions
  become real SQL UPDATE and DELETE statements.
- **Milvus** is a dedicated vector database. The collection must exist ahead of
  time with a float-vector field indexed using the cosine metric.
- **SQLite** is a zero-dependency development scaffold. It has no approximate
  nearest neighbour index: each query does a full table scan and computes cosine
  similarity in Python, returning the top results. It is perfect for local
  testing and demos because it needs no external service, but it is not intended
  for production because it scans every row on every query.

Choose SQLite to try the system on a laptop, then switch to pgvector or Milvus
for real workloads by changing only the configuration file.
