# vetosh test run (SQLite + OpenAI)

A self-contained sandbox to try the whole stack on the **SQLite scaffold** vector
store with **real OpenAI** embeddings and answers.

Requires an OpenAI key in the environment:

```bash
export OPENAI_API_KEY=sk-...
```

(To run fully offline with no key, switch `embedder` and `llm` to `type: mock` in
`config.yaml` — the commented lines.)

Run everything **from the repository root**.

## 1. Index the documents

```bash
vetosh indexer --config examples/testrun/config.yaml
```

This parses the 5 markdown files in `examples/testrun/documents/`, embeds the
chunks, and writes them into `examples/testrun/data/embeddings.db`. With
`mode: static` the indexer indexes once and exits. (You'll see a warning that the
SQLite store is a dev scaffold — that's expected.)

## 2. Start the API server

```bash
vetosh server --config examples/testrun/config.yaml
```

Retrieve chunks directly:

```bash
curl -s -X POST http://localhost:8000/retrieve \
  -H 'Content-Type: application/json' \
  -d '{"query": "How does vetosh handle deletions?", "k": 3}' | jq
```

Or the RAG endpoint (real generated answer over the retrieved context):

```bash
curl -s -X POST http://localhost:8000/rag \
  -H 'Content-Type: application/json' \
  -d '{"query": "What vector databases are supported?", "k": 3}' | jq
```

## 3. Start the web chat UI (optional)

In another terminal:

```bash
vetosh frontend --config examples/testrun/config.yaml
# open http://localhost:3000
```

## Notes

- This config uses real OpenAI: `text-embedding-3-small` for embeddings and
  `gpt-4o-mini` for `/rag` answers. Indexing the 5 sample docs is a handful of
  embedding calls (cents).
- The SQLite store has no ANN index — it linear-scans every chunk per query.
  That's fine here (a few dozen chunks); switch `vector_db` to pgvector or Milvus
  for real scale.
- To run with **no key / no network**, set `embedder` and `llm` to `type: mock`
  in `config.yaml`. The mock embedder is deterministic (non-semantic), so it
  verifies plumbing rather than relevance.
- Runtime artifacts go to `examples/testrun/data/` (git-ignored). Delete that
  folder to start fresh.
