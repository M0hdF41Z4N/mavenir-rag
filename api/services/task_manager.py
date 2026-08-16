# Redis-backed task state tracker for background ingestion jobs.
# Each task is stored as a Redis hash: "task:<task_id>" with fields:
#   status, description, progress, result (JSON), error, created_at, updated_at
# Stale tasks (still "pending"/"processing" when the server restarts) are marked
# "failed" at startup so clients don't poll forever for tasks that were killed mid-run.
import logging
from collections.abc import Callable
from datetime import UTC, datetime

from redis import Redis

from api.client.llm_client import LLMClient
from api.client.milvus_client import MilvusClient
from api.client.minio_client import MinioClient
from api.models import TopicEnum
from api.services.corpus_storage import CorpusStorageService
from api.utils.markdown_parser import MarkdownParser

logger = logging.getLogger(__name__)


class TaskManager:
    def __init__(self, redis: Redis, ttl_seconds: int = 3600):
        self.redis = redis
        self.ttl = ttl_seconds

    def create_task(self, task_id: str, description: str = "") -> None:
        key = f"task:{task_id}"
        now = datetime.now(UTC).isoformat()
        # Use a pipeline so hset and expire are sent as one atomic batch.
        # A crash between two separate calls would leave a key with no TTL.
        pipe = self.redis.pipeline()
        pipe.hset(
            key,
            mapping={
                "status": "pending",
                "description": description,
                "progress": "",
                "result": "",
                "error": "",
                "created_at": now,
                "updated_at": now,
            },
        )
        pipe.expire(key, self.ttl)
        pipe.execute()

    def update_status(
        self,
        task_id: str,
        status: str,
        progress: str | None = None,
        result: str | None = None,
        error: str | None = None,
    ) -> None:
        key = f"task:{task_id}"
        updates = {
            "status": status,
            "updated_at": datetime.now(UTC).isoformat(),
        }
        # Only include fields that were explicitly provided — avoids overwriting
        # existing values (e.g. preserving progress when only status changes).
        if progress is not None:
            updates["progress"] = progress
        if result is not None:
            updates["result"] = result
        if error is not None:
            updates["error"] = error
        # Pipeline resets TTL on every update so a long-running task doesn't expire mid-flight.
        pipe = self.redis.pipeline()
        pipe.hset(key, mapping=updates)
        pipe.expire(key, self.ttl)
        pipe.execute()

    @staticmethod
    def _compute_elapsed_time(data: dict) -> str | None:
        created = data.get("created_at")
        updated = data.get("updated_at")
        status = data.get("status")
        if not created or not updated:
            return None
        if status not in ("completed", "failed"):
            return None
        try:
            t_start = datetime.fromisoformat(created)
            t_end = datetime.fromisoformat(updated)
            delta = t_end - t_start
            total_secs = int(delta.total_seconds())
            if total_secs < 60:
                return f"{total_secs}s"
            minutes, secs = divmod(total_secs, 60)
            if minutes < 60:
                return f"{minutes}m {secs}s"
            hours, minutes = divmod(minutes, 60)
            return f"{hours}h {minutes}m {secs}s"
        except (ValueError, TypeError):
            return None

    def get_task(self, task_id: str) -> dict | None:
        key = f"task:{task_id}"
        data = self.redis.hgetall(key)
        if not data:
            return None
        data["elapsed_time"] = self._compute_elapsed_time(data)
        return data

    def list_tasks(self) -> list[dict]:
        tasks = []
        cursor = 0
        while True:
            cursor, keys = self.redis.scan(cursor=cursor, match="task:*", count=100)
            for key in keys:
                data = self.redis.hgetall(key)
                if data:
                    task_id = key.removeprefix("task:") if isinstance(key, str) else key.decode().removeprefix("task:")
                    data["task_id"] = task_id
                    data["elapsed_time"] = self._compute_elapsed_time(data)
                    tasks.append(data)
            if cursor == 0:
                break
        tasks.sort(key=lambda t: t.get("created_at", ""), reverse=True)
        return tasks

    def mark_stale_tasks_failed(self) -> None:
        cursor = 0
        stale_count = 0
        while True:
            cursor, keys = self.redis.scan(cursor=cursor, match="task:*", count=100)
            for key in keys:
                status = self.redis.hget(key, "status")
                if status in ("pending", "processing"):
                    self.redis.hset(
                        key,
                        mapping={
                            "status": "failed",
                            "error": "Server restarted during task execution",
                            "updated_at": datetime.now(UTC).isoformat(),
                        },
                    )
                    stale_count += 1
            if cursor == 0:
                break
        if stale_count:
            logger.info("Marked %d stale task(s) as failed", stale_count)


def run_ingestion_task(
    task_id: str,
    task_manager: TaskManager,
    doc_ids: list[str],
    topics: list[TopicEnum],
    minio_client: MinioClient,
    milvus_client: MilvusClient,
    llm_client: LLMClient,
    markdown_parser: MarkdownParser,
    corpus_storage: CorpusStorageService,
) -> None:
    from api.utils.rag_pipeline import rag_ingestion_pipeline

    try:
        task_manager.update_status(
            task_id,
            "processing",
            progress=f"0/{len(doc_ids)} documents processed",
        )

        progress_cb: Callable[[str], None] = lambda msg: task_manager.update_status(  # noqa: E731
            task_id, "processing", progress=msg
        )

        result = rag_ingestion_pipeline(
            doc_ids=doc_ids,
            topics=topics,
            minio_client=minio_client,
            milvus_client=milvus_client,
            llm_client=llm_client,
            markdown_parser=markdown_parser,
            corpus_storage=corpus_storage,
            progress_callback=progress_cb,
        )

        task_manager.update_status(
            task_id,
            "completed",
            progress=f"{result.processed_count}/{len(doc_ids)} documents processed",
            result=result.model_dump_json(),
        )
        logger.info(
            "Background task %s completed: %d/%d docs processed",
            task_id,
            result.processed_count,
            len(doc_ids),
        )

    except Exception as e:
        logger.error("Background task %s failed: %s", task_id, str(e))
        task_manager.update_status(task_id, "failed", error=str(e))
