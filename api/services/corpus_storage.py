# Thin service that centralises MinIO lookups for corpus objects.
# Before this existed, every router and utility called list_objects() individually with
# the same prefix pattern — this service eliminates that duplication and makes the
# "first object wins" assumption explicit in one place.
import logging

from api.client.minio_client import MinioClient
from api.models import MinioStorageObject, TopicEnum

logger = logging.getLogger(__name__)


class CorpusStorageService:
    """Handles corpus-object lookups in MinIO.

    Centralises the repeated `list_objects(prefix=corpus/<topic>/<doc_id>)` + first-item
    pattern that was duplicated across routers, scoring, and the RAG pipeline.
    """

    def __init__(self, minio_client: MinioClient) -> None:
        self.minio = minio_client

    def find_object(self, doc_id: str, topic: TopicEnum, include_metadata: bool = True) -> MinioStorageObject | None:
        """Return the first MinIO object for a corpus document, or None if not found."""
        objects = self.minio.list_objects(prefix=f"corpus/{topic.value}/{doc_id}", include_metadata=include_metadata)
        return objects[0] if objects else None

    def presigned_url_for(self, doc_id: str, topic: TopicEnum, expires_seconds: int = 3600) -> str:
        """Return a presigned GET URL for the corpus document.

        Returns an empty string if the object is not found or the presigned call fails,
        so callers do not need to guard against None.
        """
        try:
            obj = self.find_object(doc_id, topic, include_metadata=False)
            if not obj:
                return ""
            return self.minio.presigned_get_url(object_name=obj.object_name, expires_seconds=expires_seconds)
        except Exception as exc:
            logger.warning(
                "Could not generate presigned URL for doc_id=%s topic=%s: %s",
                doc_id,
                topic.value,
                exc,
            )
            return ""

    def original_filename_for(self, doc_id: str, topic: TopicEnum, fallback: str = "") -> str:
        """Return the original filename stored in MinIO metadata, or fallback."""
        obj = self.find_object(doc_id, topic, include_metadata=True)
        if not obj:
            return fallback or doc_id
        return obj.metadata.get("x-amz-meta-original-filename", fallback or doc_id)

    def object_name_for(self, doc_id: str, topic: TopicEnum) -> str | None:
        """Return the MinIO object name for a corpus document, or None if not found."""
        obj = self.find_object(doc_id, topic, include_metadata=False)
        return obj.object_name if obj else None
