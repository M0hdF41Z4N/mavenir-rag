# Background Task Processing

> **TL;DR** — Set `BACKGROUND__USE_BACKGROUND_TASKS=true` to make ingestion endpoints return a `task_id` immediately and process documents asynchronously. Poll `GET /corpus/task/{task_id}` for status. This is single-server only; no distributed queue is implemented.

When `BACKGROUND__USE_BACKGROUND_TASKS=true` is set in `.env`, the three ingestion endpoints return immediately with a `task_id` and process documents in a background thread. Task state is tracked in Redis.

## Endpoints

- `GET /corpus/tasks` — list all background tasks with status, description, progress, elapsed time
- `GET /corpus/task/{task_id}` — poll a specific task for detailed status and result

## Configuration

```env
BACKGROUND__USE_BACKGROUND_TASKS=true   # enable background processing (default: false)
BACKGROUND__TASK_RESULT_TTL_SECONDS=3600 # task result expiry in Redis (default: 1 hour)
```

## Assumptions and Limitations

### Assumptions

| #   | Assumption                                                                                                                                                                                                        | Limitation it introduces                                                                                                                                                                                              |
| --- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | **Single-server deployment.** Background threads run in the same process as the API server.                                                                                                                       | Does not scale horizontally — two server instances would spawn independent threads with no shared queue or coordination. A distributed task queue (e.g., Celery, ARQ) would be needed for multi-instance deployments. |
| 2   | **Shared clients are thread-safe.** `redis.Redis`, `Minio`, `pymilvus`, and `litellm` are assumed safe for concurrent use from multiple threads.                                                                  | Under heavy concurrent load, shared client connections could become a bottleneck or exhibit unexpected behavior. This assumption has not been stress-tested.                                                          |
| 3   | **File upload completes before backgrounding.** For `ingest-file` and `ingest-files`, the MinIO upload happens synchronously in the request; only the pipeline (parsing + embedding + insertion) is backgrounded. | Large file uploads still block the HTTP request briefly. The client must wait for the upload to finish before receiving the `task_id`.                                                                                |
| 4   | **Redis is available at startup.** The `TaskManager` attempts to clean up stale tasks on server boot.                                                                                                             | If Redis is unreachable at startup, stale cleanup is skipped (gracefully handled), but background task features will fail at runtime until Redis recovers.                                                            |
| 5   | **Daemon threads are acceptable.** Threads are set as `daemon=True` so they don't block server shutdown.                                                                                                          | If the server is stopped while a task is running, that task is killed mid-execution. It will be marked as `failed` on the next startup via stale task cleanup. No work-in-progress is recovered or resumed.           |

### Limitations

| #   | Limitation                                                                                                                                | Impact                                                                                                                                                                  |
| --- | ----------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | **No concurrency limit.** There is no cap on how many background ingestion tasks can run simultaneously.                                  | Submitting many large documents at once can exhaust CPU/memory or overwhelm the LLM/Milvus services, potentially making the server unresponsive.                        |
| 2   | **No retry mechanism.** If a background task fails (e.g., LLM timeout, Milvus connection error), it stays in `failed` status permanently. | The user must manually re-submit failed ingestion requests. There is no automatic retry or dead-letter queue.                                                           |
| 3   | **No real-time push notifications.** Clients must poll `GET /corpus/task/{task_id}` or `GET /corpus/tasks` to check progress.             | There is no WebSocket or Server-Sent Events (SSE) channel to push status updates. Clients must implement their own polling interval.                                    |
| 4   | **Task results expire.** Redis keys have a TTL (default 1 hour).                                                                          | If the client does not poll for results within the TTL window, the task data is lost. The ingested data in Milvus/MinIO is unaffected — only the task metadata expires. |
| 5   | **Per-document sequential processing.** Within a single task, documents are ingested one by one in a loop.                                | A task with 10 documents processes them sequentially, not in parallel. Total time scales linearly with document count within a single request.                          |
