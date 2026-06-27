# vetosh

**A universal, no-code RAG server for any vector database — powered by the
[Pathway](https://pathway.com) Live Data Framework.**

vetosh lets you stand up production Retrieval-Augmented Generation over your own
documents without writing code. You point it at a folder, choose a vector
database and an embedder in a YAML file (or generate one with an interactive
wizard), and run two commands:

- **`vetosh indexer`** — a Pathway streaming pipeline that watches your files,
  parses and chunks them, embeds the chunks and keeps your vector DB in sync
  (additions, modifications and deletions) in real time.
- **`vetosh server`** — a FastAPI service that embeds incoming queries and
  retrieves the most relevant chunks (and, optionally, answers them with an LLM).

The indexer and the server are fully decoupled: they share only the vector
database, so you can scale them independently.

---

## 1. Installation

vetosh requires **Python ≥ 3.10** (the minimum supported by Pathway).

```bash
pip install vetosh

# include the OpenAI client if you use OpenAI embedders / the /rag endpoint
pip install "vetosh[openai]"
```

For development:

```bash
pip install -e ".[dev,openai]"
```

---

## 2. Quickstart

The fastest way to get a valid config is the interactive wizard:

```bash
vetosh quickstart
```

It walks you through every option (each with a sensible default — press Enter to
accept), then writes a YAML file. You only *have* to provide document paths,
vector-DB connection details, API keys and your Pathway license key.

Then run the two components (in separate terminals or on separate machines):

```bash
# 1. Index your documents into the vector DB and keep it live
vetosh indexer --config config.yaml

# 2. Serve retrieval / RAG over HTTP
vetosh server --config config.yaml

# 3. (optional) Serve the web chat UI
vetosh frontend --config config.yaml   # open http://localhost:3000
```

Query the server:

```bash
curl -X POST http://localhost:8000/retrieve \
  -H 'Content-Type: application/json' \
  -d '{"query": "how does persistence work?", "k": 5}'
```

```jsonc
{
  "results": [
    {"text": "...", "metadata": {"path": "/data/docs/guide.pdf", ...}, "score": 0.95}
  ]
}
```

If you configured an `llm` section, the `/rag` endpoint also answers questions:

```bash
curl -X POST http://localhost:8000/rag \
  -H 'Content-Type: application/json' \
  -d '{"query": "summarize the onboarding guide", "k": 5}'
```

```jsonc
{ "answer": "...", "sources": [ {"text": "...", "metadata": {...}, "score": 0.91} ] }
```

---

## 3. Pathway License Key

Pathway requires a **free** license key. Get yours in one click at
**<https://pathway.com/framework/get-license>** (email or LinkedIn sign-in).

Provide it either via the config file:

```yaml
pathway_license_key: ${PATHWAY_LICENSE_KEY}
```

or directly as a string. Using the `${PATHWAY_LICENSE_KEY}` form keeps the key
out of the file — set it in the environment:

```bash
export PATHWAY_LICENSE_KEY="your-key-here"
```

The key is applied when the indexing graph is initialized.

---

## 4. Config reference

A single YAML file can configure both the indexer and the server (a *universal*
config), or you can split it into an indexer-only and a server-only file. Any
string value supports `${ENV_VAR}` interpolation, so credentials never need to be
hardcoded.

| Field | Type | Default | Used by | Description |
|---|---|---|---|---|
| `pathway_license_key` | str | — | indexer | Free Pathway license key (or `${ENV}`). |
| `sources` | list | — (required) | indexer | One or more sources to index (mixed types allowed). |
| `sources[].type` | `fs`\|`gdrive` | `fs` | indexer | Source type — see [Sources](#sources) below. |
| `sources[].mode` | `streaming`\|`static` | `streaming` | indexer | `streaming` watches continuously; `static` indexes once and exits. |
| **fs** `sources[].path` | str | — (required) | indexer | Directory to watch. |
| **fs** `sources[].glob` | str | `**/*` | indexer | Glob of files to include. |
| **gdrive** `sources[].object_id` | str | — (required) | indexer | Drive folder or file id (folders scanned recursively). |
| **gdrive** `sources[].service_user_credentials_file` | str | — (required) | indexer | Path to a Google service-account JSON credentials file. |
| **gdrive** `sources[].file_name_pattern` | str\|list | — | indexer | Optional file-name glob(s), e.g. `*.pdf`. |
| **gdrive** `sources[].object_size_limit` | int | — | indexer | Optional max object size in bytes. |
| `vector_db.type` | `pgvector`\|`milvus` | — (required) | both | Vector database backend. |
| `vector_db.connection_string` | str | — | both (pgvector) | `postgresql://user:pass@host/db`. |
| `vector_db.table` | str | `vetosh_embeddings` | both (pgvector) | Target table. |
| `vector_db.uri` | str | — | both (milvus) | Milvus URI, e.g. `http://localhost:19530`. |
| `vector_db.host` / `vector_db.port` | str / int | `localhost` / `19530` | both (milvus) | Used if `uri` is omitted. |
| `vector_db.collection` | str | `vetosh_embeddings` | both (milvus) | Milvus collection. |
| `embedder.type` | str | `openai` | both | Embedder family (see below). |
| `embedder.model` | str | provider default | both | Model name. |
| `embedder.api_key` | str | — | both | API key (or `${ENV}`). |
| `splitter.type` | `token_count`\|`recursive` | `token_count` | indexer | Chunking strategy. |
| `splitter.chunk_size` | int | `512` | indexer | Max chunk size (tokens). |
| `splitter.chunk_overlap` | int | `50` | indexer | Overlap (used by `recursive`). |
| `persistence.enabled` | bool | `true` | indexer | See [Persistence](#5-persistence). |
| `persistence.backend` | `filesystem` | `filesystem` | indexer | Persistence backend. |
| `persistence.path` | str | `/var/vetosh/persistence` | indexer | Persistence directory. |
| `server.host` / `server.port` | str / int | `0.0.0.0` / `8000` | server | Bind address. |
| `llm.type` | str | `openai` | server | LLM for `/rag` (omit to disable `/rag`). |
| `llm.model` | str | `gpt-4o-mini` | server | Chat model. |
| `llm.api_key` | str | — | server | API key (or `${ENV}`). |
| `frontend.host` / `frontend.port` | str / int | `0.0.0.0` / `3000` | frontend | Bind address for the chat UI. |
| `frontend.api_url` | str | `http://localhost:8000` | frontend | Base URL of the API the frontend proxies to. |
| `frontend.title` | str | `vetosh` | frontend | Title shown in the chat UI. |

**Supported embedders** (from `pathway.xpacks.llm.embedders`): `openai`,
`litellm`, `sentence_transformer`, `gemini`, `bedrock`. The server embeds queries
through an OpenAI-compatible async client for `openai`/`litellm`; make sure the
indexer and server use the **same** embedder model so vectors are comparable.

### Sources

vetosh only supports sources that offer Pathway's **`only_metadata`** read mode —
the graph holds just paths/identifiers and metadata (never the file bytes), and
the bytes are fetched on demand during parsing. That keeps the memory footprint
tiny (1 GB of PDFs stays a few KB in the pipeline). Today that means:

- **`fs`** — a local directory, watched recursively per `glob`.
- **`gdrive`** — a Google Drive folder or file. Create a Google **service
  account**, download its JSON key, share the target folder with the service
  account's email, and point `service_user_credentials_file` at the key. See the
  [Pathway Google Drive connector guide](https://pathway.com/developers/user-guide/connect/connectors/gdrive-connector/).
  Install the extra: `pip install "vetosh[gdrive]"`.

You can list several sources of mixed types. **S3/MinIO** (no `only_metadata`
mode) and **SharePoint** (no connector in the current Pathway build) are not
supported yet.

### pgvector table

The indexer writes in *snapshot* mode keyed by a chunk id, so updates and
deletions are reflected as real `UPDATE`/`DELETE`s. Create the target table with
a `vector(n)` column sized to your embedder's dimension, e.g. for
`text-embedding-3-small` (1536 dims):

```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE vetosh_embeddings (
  chunk_id  text PRIMARY KEY,
  text      text,
  metadata  jsonb,
  embedding vector(1536)
);
```

(The indexer writes the chunk primary key to the `chunk_id` column.)

### Milvus collection

Create the collection ahead of time with a primary key field `id` (VARCHAR), a
`text` (VARCHAR) field, a `metadata` (JSON) field and an `embedding`
(FLOAT_VECTOR) field of the right dimension, indexed with the `COSINE` metric.

---

## 5. Persistence

When `persistence.enabled: true`, the indexer uses Pathway's built-in
persistence backend plus a `DefaultCache` UDF call-cache on the embedder. On
restart, Pathway replays its persisted state and re-emits only the *diffs* since
the last run, so:

- **unchanged documents are not re-embedded** (the embedder cache and operator
  state are restored), and
- **documents removed while the indexer was down are correctly retracted** from
  the vector DB on the next run.

This is independent of vetosh's separate, persistent *parse cache* (see
[Architecture](#7-architecture)), which keeps extracted document text warm across
restarts and is evicted when a document is removed so it stays bounded.

Disabling persistence still produces identical vectors for a given set of files;
it only forgoes the cross-restart diffing and caching.

---

## 6. Unsupported formats

vetosh never implements its own parsers — it dispatches each file to a
`pathway.xpacks.llm.parsers` parser by extension:

- `.pdf` → `PypdfParser`
- `.txt`, `.md`, `.markdown`, `.text` → `Utf8Parser`
- everything else → `UnstructuredParser` (DOCX, PPTX, XLSX, HTML, EML, CSV, RTF,
  EPUB and the other formats supported by [unstructured](https://docs.unstructured.io/))

Not supported out of the box:

- **Images** (`.png`, `.jpg`, …) and **audio/video** — these require the vision
  (`ImageParser`/`SlideParser`) or Whisper (`AudioParser`) parsers, which need a
  multimodal model and are not wired into the default dispatch.
- **Encrypted / password-protected PDFs.**
- **Google-native files** (Google Docs/Sheets/Slides) from a `gdrive` source —
  they have no direct binary download and would need an export step (TODO).
  Regular uploaded files (PDF, DOCX, …) in Drive work fine.
- Any format not handled by the `unstructured` library version you have
  installed.

Unsupported files are skipped (they simply produce no chunks).

---

## 7. Architecture

```
                 ┌──────────────────────── vetosh indexer (Pathway) ───────────────────────┐
                 │                                                                          │
   files on disk │   fs.read(only_metadata)   parse UDF        splitter      embedder       │
  ┌───────────┐  │   ┌──────────────────┐   ┌───────────┐   ┌──────────┐   ┌──────────┐    │
  │ /data/... │──┼──▶│ paths + metadata │──▶│ extract   │──▶│  chunk   │──▶│ vectors  │──┐ │
  └───────────┘  │   │ (no file bytes)  │   │ text      │   │          │   │          │  │ │
                 │   └──────────────────┘   └─────┬─────┘   └──────────┘   └──────────┘  │ │
                 │                                │ persistent parse cache                │ │
                 │                                │ (keyed by metadata, evicted on        │ │
                 │                                │  removal via is_addition)             │ │
                 └────────────────────────────────┼──────────────────────────────────────┼─┘
                                                  │                                      │
                                                  ▼                                      ▼
                                         ┌──────────────────────────────────────────────────┐
                                         │              Vector database                      │
                                         │            (pgvector / Milvus)                    │
                                         └──────────────────────────────────────────────────┘
                                                  ▲
                                                  │ top-k nearest chunks
                 ┌────────────────────────────────┼──────────────────────────────────────┐
                 │   POST /retrieve, /rag         │                  vetosh server         │
   client  ──────┼──▶ embed query ────────────────┘  (FastAPI, async, horizontally        │
                 │                                     scalable, decoupled from indexer)   │
                 └───────────────────────────────────────────────────────────────────────┘
```

**Why `only_metadata`?** The filesystem connector only ever puts *paths and
metadata* into the graph, never file contents — so 1 GB of PDFs costs a few KB in
the pipeline. Text is extracted on demand inside the parse UDF.

**Deletion handling.** The parse UDF is non-deterministic, so Pathway memoizes
its output and, when a file is removed, re-emits that stored text *negated*
without re-reading the (now gone) file; the retraction flows through chunking and
embedding and the sink deletes exactly the matching vectors. A
`pw.io.subscribe` side-channel uses the native per-record `is_addition` flag to
evict the persistent parse-cache entry, keeping the cache bounded.

---

## 8. Scaling

The indexer and the server are **separate processes that share only the vector
database**:

- **Frontend.** A separate, stateless web tier (`vetosh frontend`) that serves
  the chat UI and proxies to the API at `frontend.api_url`. Run it on its own
  worker/host; point several instances at one (or a pool of) API endpoints.
- **Server.** Stateless and read-only against the vector DB — run as many
  instances as you like behind a load balancer, across machines or availability
  zones. Because it is fully async (coroutines, not a thread pool), each instance
  handles many concurrent network-bound retrieval calls efficiently.
- **Indexer.** Runs as its own long-lived process and can be scaled with
  Pathway's multi-worker support; its persistence lets it resume without
  re-embedding.
- **Vector database.** Scaling and replicating the vector DB itself (pgvector or
  Milvus) is the user's responsibility and follows that database's own guidance.
```
