# Orchestrates the two-phase ingestion flow: upload to MinIO, then parse + embed + index.
# Supports both synchronous execution (for testing / small loads) and FastAPI
# BackgroundTasks (for production) — controlled by BACKGROUND__USE_BACKGROUND_TASKS.
import logging
import uuid
from pathlib import Path

from fastapi import BackgroundTasks, UploadFile

from api.client.llm_client import LLMClient
from api.client.milvus_client import MilvusClient
from api.client.minio_client import MinioClient
from api.config import Settings
from api.models import (
    RAGIngestionResponse,
    TaskAcceptedResponse,
    TaskStatusEnum,
    TopicEnum,
)
from api.services.corpus_storage import CorpusStorageService
from api.services.task_manager import TaskManager, run_ingestion_task
from api.utils.markdown_parser import MarkdownParser

logger = logging.getLogger(__name__)


class IngestionService:
    """Consolidates file upload, pipeline dispatch, and background task creation.

    Replaces the repeated upload + rag_ingestion_pipeline + threading.Thread
    pattern that was duplicated across the three ingest endpoints.
    """

    def __init__(
        self,
        minio_client: MinioClient,
        milvus_client: MilvusClient,
        llm_client: LLMClient,
        markdown_parser: MarkdownParser,
        task_manager: TaskManager,
        corpus_storage: CorpusStorageService,
        settings: Settings,
    ) -> None:
        self.minio = minio_client
        self.milvus = milvus_client
        self.llm = llm_client
        self.markdown_parser = markdown_parser
        self.task_manager = task_manager
        self.corpus_storage = corpus_storage
        self.settings = settings

    async def upload_files(self, files: list[UploadFile], topic: TopicEnum) -> list[str]:
        """Upload files to MinIO and return their generated doc_ids."""
        doc_ids = []
        for file in files:
            doc_id = str(uuid.uuid4())
            ext = Path(file.filename).suffix
            object_name = f"corpus/{topic.value}/{doc_id}/{ext}"
            content = await file.read()
            self.minio.upload_bytes(
                data=content,
                object_name=object_name,
                metadata={"original-filename": file.filename},
            )
            doc_ids.append(doc_id)
            logger.info(
                "IngestionService.upload_files: uploaded '%s' as doc_id=%s",
                file.filename,
                doc_id,
            )
        return doc_ids

    def start(
        self,
        doc_ids: list[str],
        topics: list[TopicEnum],
        description: str,
        background: BackgroundTasks | None = None,
    ) -> RAGIngestionResponse | TaskAcceptedResponse:
        """Run the ingestion pipeline synchronously or as a FastAPI background task.

        Behaviour is controlled by `BACKGROUND__USE_BACKGROUND_TASKS` in settings.
        When background processing is enabled, `background` (FastAPI BackgroundTasks)
        must be provided by the calling route handler.
        """
        if not self.settings.background.use_background_tasks:
            from api.utils.rag_pipeline import rag_ingestion_pipeline

            return rag_ingestion_pipeline(
                doc_ids=doc_ids,
                topics=topics,
                minio_client=self.minio,
                milvus_client=self.milvus,
                llm_client=self.llm,
                markdown_parser=self.markdown_parser,
                corpus_storage=self.corpus_storage,
            )

        task_id = str(uuid.uuid4())
        self.task_manager.create_task(task_id, description=description)
        # Use FastAPI BackgroundTasks so the task lifecycle is managed by the
        # ASGI server rather than raw daemon threads, avoiding thread leaks on shutdown.
        background.add_task(
            run_ingestion_task,
            task_id,
            self.task_manager,
            doc_ids,
            topics,
            self.minio,
            self.milvus,
            self.llm,
            self.markdown_parser,
            self.corpus_storage,
        )
        logger.info(
            "IngestionService.start: background task %s queued for %d doc(s)",
            task_id,
            len(doc_ids),
        )
        return TaskAcceptedResponse(
            task_id=task_id,
            doc_ids=doc_ids,
            status=TaskStatusEnum.PENDING,
            poll_url=f"/corpus/task/{task_id}",
        )
