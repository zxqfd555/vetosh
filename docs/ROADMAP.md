# vetosh — status & roadmap

*Last updated: 2026-07-02.*

## Where we are

vetosh is a no-code RAG stack built as a **showcase of Pathway connectors**:
one YAML file turns Pathway's native `pw.io.*` output connectors + `xpacks.llm`
parsers/splitters/embedders into a live document-indexing pipeline, a retrieval
API and a chat UI — with zero application code written by the user.

**Vector-DB backends (8):** DuckDB (embedded, zero-setup default), pgvector,
Milvus, Qdrant, ChromaDB, Weaviate, Pinecone, MongoDB Atlas Vector Search.
All written through native Pathway connectors in snapshot mode — source
adds/edits/deletes become real upserts/deletes in the store.

**Embedders (5 families, both sides):** OpenAI, LiteLLM (≈100 providers),
SentenceTransformers (fully local), Gemini, Bedrock.

**Modalities today:** text (`.txt`/`.md`), PDF (pypdf), and everything
`unstructured` handles (DOCX, PPTX, XLSX, HTML, EML, CSV, RTF, EPUB, …).
Available in the xpack but not yet wired in: images/slides via vision LLMs
(`DoclingParser`, `ImageParser`, `SlideParser`), OCR (`PaddleOCRParser`),
audio via Whisper (`AudioParser`). No video.

**The differentiator**: *continuous incremental sync with correct deletion
semantics into an external vector DB of your choice*. Competing tools re-index
on a schedule or on upload; vetosh watches sources and propagates a file edit
or removal to the vector DB within seconds, without re-embedding unchanged
content (Pathway persistence + embedder cache). That story is exactly one YAML
file — because the machinery is Pathway itself.

## Competitive landscape (as of 2026-07)

| Tool | Stars | External vector DBs | Continuous sync incl. deletes |
|---|---|---|---|
| **vetosh** | — | **8, native connectors** | **yes — streaming watch, real deletes** |
| Dify | 147k | ~20, but captive KB storage | no (manual / triggered re-sync) |
| RAGFlow | 84k | none (bundled ES/Infinity) | no (batch upload) |
| AnythingLLM | 62k | ~10 | partial beta: per-file hourly re-embed, desktop-only |
| PrivateGPT | 57k | ~5 | no (bulk ingest; dormant) |
| LlamaIndex OSS | 51k | 40+ | deletes only on manual pipeline re-run |
| Onyx (ex-Danswer) | 31k | none (captive Vespa→OpenSearch) | closest: 30-min polling + periodic pruning |
| Haystack | 26k | ~27 | no (DIY pipelines, batch) |
| Airbyte vector dest. | 22k | ~7 | scheduled; deletes only from CDC DB sources |
| Unstructured | 15k | ~20 | scheduled incremental; no delete propagation |
| R2R | 8k | 1 (pgvector) | no (API-driven) |
| LlamaCloud (SaaS) | — | 7 sinks | near-match, but proprietary/managed |

The open slot vetosh occupies: **OSS, watches folders/Drive continuously,
propagates adds+edits+deletes in seconds, and writes into *your* vector DB**
— either captive-index tools (Onyx, RAGFlow, Dify) or batch external-DB
writers (Airbyte, Unstructured, LlamaIndex) cover only one half each. The
pitch to lead with: *"the only open-source tool where deleting a file
actually deletes its vectors — in any of 8 databases — live."*

**Known constraints:**

- The new connectors (duckdb/qdrant/chroma/weaviate/pinecone) need a Pathway
  release that includes them; today vetosh runs against the develop tree.
- DuckDB is single-writer: a *streaming* indexer excludes a concurrently
  serving reader process — use `mode: static` with DuckDB, or a client-server
  backend for live sync.
- pgvector/Milvus/Chroma/Weaviate/Pinecone targets must be pre-created by the
  user (dimension mismatch surfaces only at write time).

---

## Category 1 — before showing to business colleagues

*Goal: a flawless 5-minute "look what falls out of Pathway connectors for
free" demo. Polish the first-run experience; make the live-sync magic visible.*

1. ~~**`vetosh up`**~~ — DONE: a dumb supervisor that spawns `vetosh indexer`
   and `vetosh server` as children simultaneously (uniform streaming for all
   backends, no refresh loops) and tears both down on exit; if the indexer
   finishes (static sources) the server keeps serving. DuckDB + streaming
   currently warns about the single-writer lock — resolves when Pathway ships
   `detach_between_batches` for `pw.io.duckdb.write` (requested; the vetosh
   accessor already retries through brief lock windows).
2. **Bundled demo corpus + script** (`vetosh demo`): sample docs, a scripted
   moment where a file is edited/deleted and the chat answer visibly changes —
   that moment *is* the technology pitch.
3. **Fully local no-key path as the wizard default**: DuckDB +
   `sentence_transformer` — nothing to sign up for except the free Pathway
   license.
4. **docker-compose for the live-sync demo** (qdrant + vetosh services), so
   the "edit a file, watch the answer change" scene runs on a client-server
   backend.
5. **Re-record the README GIF** for the new flow (DuckDB default, new wizard).
6. **Friendly failure modes**: missing license key, unreachable DB, embedder
   dimension mismatch — each should produce one clear sentence, not a stack
   trace. Add `vetosh doctor` (validate config, ping DB, check embedder
   dimension vs. index).
7. **Index statistics endpoint** (`/stats`: chunk/document counts, last update
   time) surfaced in the frontend — makes liveness visible.
8. **Smoke-test the demo path in CI** so the show never breaks: quickstart →
   index → retrieve on DuckDB.

## Category 2 — before showing to the external world

*Goal: credibility. Anyone can install it, run it against their own stack, and
nothing embarrasses us.*

1. **Depend on a released Pathway** that ships the new connectors (today:
   develop build). Blocks everything — PyPI users cannot run the new backends.
2. **Integration-test matrix for all 8 backends** (docker-compose services in
   CI; Pinecone Local for pinecone; Atlas local dev image for mongodb).
3. **PyPI release automation** + versioning + changelog; `pip install vetosh`
   must just work.
4. **Server hardening**: optional API-key auth, CORS config, rate limiting,
   request size limits, timeouts.
5. **Auto-create targets where possible**: pgvector table (dimension from
   `BaseEmbedder.get_embedding_dimension()`), Milvus collection, Pinecone
   index — remove the biggest first-run trap (config promises
   `embedding_dimension` but nothing consumes it today).
6. **Observability**: structured logs, `/metrics` (Prometheus), indexing lag
   gauge.
7. **Streaming `/rag` responses** (SSE) in server + frontend.
8. **More modalities via config**: `parser:` section exposing Docling
   (PDF+tables+images), ImageParser, AudioParser, PaddleOCR from the xpack.
9. **S3 / SharePoint / pyfilesystem sources** — `only_metadata` support just
   landed in Pathway develop (#10483); wire them into `sources:`.
10. **Optional reranker** (xpack `rerankers`) and hybrid retrieval where the
    backend supports it.
11. **Docs site**: per-backend tutorials, troubleshooting, architecture page;
    CONTRIBUTING + issue templates; choose and state the OSS license clearly.
12. **CI quality gates**: ruff, mypy, py3.10–3.13 matrix.

## Category 3 — after the public prototype

1. **MCP server mode** (`pathway.xpacks.llm.mcp_server`): "chat with your
   docs from Claude/IDE" — big reach, low effort.
2. **Adaptive RAG** (geometric context growth, from
   `AdaptiveRAGQuestionAnswerer`) as an opt-in `/rag` strategy.
3. **Evaluation harness** (RAGAS or similar) + per-backend latency/recall
   benchmarks; publish numbers.
4. **LEANN backend** once its reader path matures (today: write-only,
   full rebuild per commit).
5. **DuckDB ANN**: upstream a `FLOAT[n]` fixed-size array option in
   `pw.io.duckdb.write` so the VSS/HNSW index applies; today retrieval is a
   vectorized in-DB scan (fine at demo scale).
6. **Multi-collection / multi-tenant** serving; per-user corpora.
7. **Frontend growth**: chat history, source previews, auth.
8. **Kubernetes/Helm deployment** recipes; horizontal-scale guide.
9. **Video modality** (frame sampling + vision parsing) — no xpack support
   yet; needs design.
