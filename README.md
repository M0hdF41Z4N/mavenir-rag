# Mavenir Rag

Mavenir Rag is a FastAPI-based RAG service for:

- ingesting corpus documents (PDF/DOCX),
- storing embeddings in Milvus with metadata,
- storing source artifacts/pages/images in MinIO,
- running similarity search against the indexed corpus, and
- generating LLM-based questions from matched content.

## 1. Tech Stack

- Python 3.13+
- FastAPI + Uvicorn
- LiteLLM (chat + embeddings; supports Ollama/OpenAI/Gemini)
- Milvus (vector database)
- MinIO (object storage)
- Poetry (dependency + script management)

## 2. Project Structure

- `api/` - FastAPI app, routers, clients, and RAG utilities
- `frontend/` - Streamlit chat UI (separate process, talks to the API over HTTP)
- `tests/` - unit/integration-style tests for clients
- `docker-compose.yml` - local Milvus + MinIO + etcd stack
- `Dockerfile` - API container build
- `.env.example` - environment variable template

## 3. Prerequisites

- Python `3.13.x` (developed on 3.13.5)
- Poetry `2.1.3` — older versions may break this project ([install guide](https://python-poetry.org/docs/))
- Docker + Docker Compose
- Ollama installed locally (if using default Ollama provider)

## 4. Setup

1. Install dependencies:

```bash
poetry install
```

2. Configure environment:

```bash
cp .env.example .env
```

Then update `.env` values for your machine. In this repo's local Docker setup:

- Milvus host port is mapped to `19531` (`MILVUS__PORT=19531`)
- MinIO S3 API is mapped to `9002` (`MINIO__ENDPOINT=localhost:9002`)
- MinIO console is exposed on `9003`

## 5. Run Locally

Start backing services (Milvus + MinIO + Ollama models):

```bash
poetry run setup:all
```

Or start individually:

```bash
poetry run setup:milvus-minio
poetry run setup:ollama
```

Start API server:

```bash
poetry run start
```

The API runs on `http://localhost:8001` and docs are available at:

- `http://localhost:8001/docs`
- `http://localhost:8001/redoc`

## 6. Stop Services

```bash
poetry run stop:all
```

Or individually:

```bash
poetry run stop:ollama
poetry run stop:milvus-minio
```

To stop all services **and permanently delete all persisted data** (Milvus collections, MinIO buckets, etcd state, Redis data):

```bash
poetry run clear:data
```

> **Warning:** This is irreversible. All ingested documents, embeddings, and stored objects will be lost.

## 7. Poetry Scripts

- `poetry run start` - run API with reload
- `poetry run serve` - run API without reload
- `poetry run test` - run tests + coverage
- `poetry run setup:ollama`
- `poetry run setup:milvus-minio`
- `poetry run setup:all`
- `poetry run stop:ollama`
- `poetry run stop:milvus-minio`
- `poetry run stop:all`
- `poetry run clear:data` — stop all services and delete all persisted volume data

## 8. Features

What the system gives you, and the mechanism behind each one.

### Answer trustworthiness

- **Answers stay tied to your documents.** Retrieval fuses three signals, then an optional second LLM pass splits the answer into atomic claims and checks each one against the retrieved chunks (`api/services/groundedness_verifier.py`). A separate judging pass catches confident fabrications that a generation prompt alone cannot.
- **Abstains rather than guessing.** If nothing is retrieved, or the best fused score falls under `CHAT__MIN_EVIDENCE_SCORE`, the endpoint returns a fixed abstention without calling the LLM at all. When the verifier finds unsupported claims, `abstain_on_ungrounded` replaces the answer instead of shipping a partly invented one.
- **Every claim points back to a page.** Retrieved chunks are fed to the model as numbered `[n]` blocks, so verdicts map each claim to the chunk, file, and page number it came from.
- **Source disagreement is surfaced, not hidden.** A contradiction pass scans retrieved chunks for mutually exclusive claims and reports the conflicting pairs (`api/services/contradiction_detector.py`), instead of silently answering from whichever chunk ranked higher.
- **Checks fail closed.** If a verifier LLM call or JSON parse fails, it reports `checked=False` and unverified, so a broken check never reads as a passed one.

### Retrieval quality

- **Matches both meaning and exact wording.** Milvus hybrid search runs a dense COSINE vector (HNSW) alongside BM25 sparse retrieval, so paraphrases and literal strings like acronyms and error codes both land. Fusion uses `WeightedRanker`.
- **Section headings count for more than body text.** BM25 runs over two fields, chunk body and heading path, with heading weighted highest (`heading_bm25_weight=0.5`), so a query hitting a section title outranks an incidental body mention.
- **Sharper final shortlist.** An optional cross-encoder pass (`api/services/reranker.py`) over-fetches a wide candidate pool and re-scores each query and chunk pair read together, then keeps the top-k. It fails safe: on any error the original order is returned.
- **One excellent match is enough to rank a document.** Document scores default to `max` over chunk scores, so a pile of weak chunks cannot dilute a document that has one strong hit (`api/utils/scoring.py`).
- **Topics stay isolated.** Every search carries a mandatory topic filter, so a query never crosses into another topic's corpus.

### Document understanding

- **Tables and diagrams become searchable text.** Two parsers are combined: `pymupdf4llm` for fast page-wise markdown, and `docling` for real table structure and figures. Diagrams are described by a vision LLM and the description is written into the chunk.
- **Wasted vision calls are filtered out.** Images under 150x150 px are skipped outright, and the description prompt returns `SKIP` for content-free blobs, which is validated before anything is embedded.
- **Chunks keep their section context and page number.** Splitting is header-aware, each chunk carries its ordered heading ancestry plus `page_num`, and oversized chunks are cut on paragraph and code-block boundaries under a `tiktoken` budget.
- **Image and table questions work with no extra plumbing.** Descriptions and table markdown already live inside the ingested chunks, so the same retrieval path answers them.

### Operations

- **Large ingests do not block the caller.** Ingestion returns a `task_id` immediately and runs in a background thread with progress tracked in Redis. Poll `GET /corpus/tasks`. See [docs/background-task-processing.md](docs/background-task-processing.md) for the concurrency limits and trade-offs.
- **Conversations survive reconnects.** Chat history and sessions are Redis-backed, and only the recent `history_context_k` turns are sent as LLM context to keep prompts small.
- **Four question formats from one corpus.** Subjective, MCQ, one-word, and match-making, selected per request by `question_answer_type`.
- **Shareable output.** Each inference run writes a markdown report to MinIO and returns a presigned URL.

> **Off by default.** Groundedness checking, contradiction detection, and cross-encoder reranking each cost an extra LLM call or a model load, so they ship disabled. Turn them on with `CHAT__ENABLE_GROUNDEDNESS_CHECK`, `CHAT__ENABLE_CONTRADICTION_CHECK`, and `RERANK__ENABLE_RERANK`. The evidence floor `CHAT__MIN_EVIDENCE_SCORE` defaults to `0.0`, which only abstains when retrieval comes back empty, so raise it per corpus to make the gate bite. Background tasks are on by default.

## 9. File Handling — Assumptions and Limitations

See [docs/file-handling.md](docs/file-handling.md) for the full breakdown of assumptions, memory/disk limitations, and possible optimizations.

## 10. Testing

Run the unit test suite:

```bash
poetry run test
```

Coverage HTML output is generated in `python_coverage/`.

### E2E Integration Test

The E2E test drives the full RAG pipeline against a live API:
**Ingest → Poll task → Generate questions → Chatbot Q&A → HTML reports**

**Prerequisites:** API server and backing services must be running.

```bash
# Terminal A — start backing services
poetry run setup:all

# Terminal B — start API
poetry run start

# Terminal C — run E2E test
poetry run python -m tests.e2e.runner
```

On success the runner prints two clickable `file://` links:

```
Summary:  file:///abs/path/tests/reports/<timestamp>/summary.html
Detailed: file:///abs/path/tests/reports/<timestamp>/detailed.html
```

Open either file in a browser. The **summary** shows one Q+A excerpt per question; the **detailed** report has the full chat transcripts, source tables, and the raw infer response in a collapsible block.

To run via the existing `poetry run test` discovery (requires a live API):

```bash
RUN_E2E=1 poetry run test
```

Without `RUN_E2E=1` the test is automatically skipped so the unit suite stays offline.

**Configuration** (edit `tests/e2e/config.py`):

| Variable | Default | Description |
|---|---|---|
| `BASE_URL` | `http://localhost:8001` | API base URL |
| `TOPIC` | `general` | Topic used for ingestion and search |
| `POLL_INTERVAL_SEC` | `20` | Seconds between task status polls |
| `POLL_TIMEOUT_SEC` | `900` | Max wait time for ingestion (15 min) |
| `CHAT_QUESTION_LIMIT` | `5` | Max questions sent to the chatbot |

## 11. Pre-Deployment Checks

Run through this checklist before deploying to any non-local environment.

| #   | Check                                | How to verify                                                                                                                                                                  |
| --- | ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 1   | **CORS origins locked down**         | Set `SECURITY__CORS_ALLOWED_ORIGINS=["https://your-frontend.example.com"]` in `.env`. Never deploy with the default `["*"]`.                                                   |
| 2   | **Exception detail hidden**          | Set `SECURITY__EXPOSE_EXCEPTION_DETAILS=false`. The global error handler will return only `{"detail": "Internal Server Error"}` — full tracebacks are logged server-side only. |
| 3   | **Credentials from secrets**         | Confirm `MINIO__ACCESS_KEY`, `MINIO__SECRET_KEY`, Redis password (if any), and LLM API keys come from `.env` or a secrets manager — not the dev defaults (`minioadmin`).       |
| 4   | **Reload disabled**                  | Use `poetry run serve` (calls `uvicorn` without `reload=True`) or a gunicorn/uvicorn production process manager. Never use `poetry run start` in production.                   |
| 5   | **Log level set to INFO**            | Confirm `LOG_LEVEL=INFO` in your process environment. `DEBUG` logs may expose request payloads.                                                                                |
| 6   | **MinIO endpoint is not localhost**  | Set `MINIO__ENDPOINT` to the correct host:port for the target environment (not `localhost:9002`).                                                                              |
| 7   | **Milvus host and port are correct** | Set `MILVUS__HOST` and `MILVUS__PORT` for the target environment.                                                                                                              |

> **Future improvement (low priority):** Add per-request correlation IDs to log records for easier debugging under concurrent load. This requires a small `contextvars`-based logging middleware and is tracked as a future improvement.

## 12. Future Enhancements

Remaining gaps between the current implementation and the target RAG architecture. See `TODO.md` for the tracked task list.

| # | Enhancement | Current state | Notes |
| --- | --- | --- | --- |
| 1 | **DOCX / TXT ingestion** | Config allows `.pdf/.docx/.txt/.md` (`api/config.py`), but the parser hardcodes PDF (`pymupdf.open(..., filetype="pdf")` in `api/utils/parsers/hybrid_parser.py`; only `InputFormat.PDF` is registered with docling). | Add real per-format parsing so DOCX/TXT are extracted correctly instead of failing or being misparsed. |
| 2 | **RRF fusion option** | Milvus hybrid search fuses dense + BM25 with `WeightedRanker` only (`api/client/milvus_client.py`). A cross-encoder reranker (`api/services/reranker.py`) already runs as a separate second-stage pass. | Add Milvus `RRFRanker` as a selectable fusion strategy alongside the weighted ranker. |
| 3 | **HuggingFace embedding provider** | Provider enum supports `ollama` / `openai` / `gemini` (`api/config.py`). | Add a HuggingFace / sentence-transformers dense-embedding option. Claude can be plugged in as a chat provider via LiteLLM (`anthropic/<model-id>`), but has no embedding endpoint, so embeddings stay on Ollama/OpenAI/HF. |
