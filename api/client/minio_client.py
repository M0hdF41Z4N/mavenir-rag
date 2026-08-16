# Object storage client. All corpus documents, extracted page images, and generated
# markdown reports are stored and retrieved through this wrapper. Object paths follow
# the convention:  <prefix>/<topic>/<doc_id>/<filename>
from datetime import timedelta
from io import BytesIO

from minio import Minio
from minio.deleteobjects import DeleteObject

from api.config import MinioSettings
from api.models import (
    DeleteObjectsByDocIDsResponse,
    DocumentIDTopicPair,
    MetadataSchema,
    MinioStorageObject,
)


class MinioClient:
    def __init__(
        self,
        settings: MinioSettings,
    ) -> None:
        """
        Initialize MinIO client using Settings object.
        """
        self.config = settings
        self.bucket_name = self.config.bucket_name

        self.client = Minio(
            self.config.endpoint,
            access_key=self.config.access_key,
            secret_key=self.config.secret_key,
            secure=self.config.secure,
        )

        # SigV4 includes Host in the canonical request, so presigned URLs must be
        # signed against the host the browser will resolve. When public_endpoint
        # is unset, sign with the internal client (dev / single-host setups).
        #
        # region is pinned so the SDK skips its `GET /<bucket>?location=` region-
        # discovery call. That call would otherwise hit the public endpoint from
        # inside the LAN and fail on hairpin NAT. Set MINIO__REGION to match the
        # server's MINIO_REGION; defaults to us-east-1 (MinIO's own default).
        if self.config.public_endpoint:
            self.signer = Minio(
                self.config.public_endpoint,
                access_key=self.config.access_key,
                secret_key=self.config.secret_key,
                secure=self.config.public_secure,
                region=self.config.region or "us-east-1",
            )
        else:
            self.signer = self.client

        self._ensure_bucket_exists()

    # --------------------------------------------------

    def _ensure_bucket_exists(self) -> None:
        """Create bucket if it doesn't exist."""
        if not self.client.bucket_exists(self.bucket_name):
            self.client.make_bucket(self.bucket_name)

    # --------------------------------------------------

    def upload_file(
        self,
        file_path: str,
        object_name: str,
        metadata: MetadataSchema | None = None,
    ) -> None:
        """
        Upload a file to MinIO with an explicit object name and optional metadata.

        Args:
            file_path: Local file path
            object_name: Unique name (including path) in bucket (e.g. 'corpus/uuid.pdf')
            metadata: Custom metadata (e.g. {'original-filename': 'manual.pdf'})
        """
        self.client.fput_object(
            bucket_name=self.bucket_name,
            object_name=object_name,
            file_path=file_path,
            metadata=metadata,
        )

    # --------------------------------------------------

    def upload_bytes(
        self,
        data: bytes,
        object_name: str,
        content_type: str = "application/octet-stream",
        metadata: MetadataSchema | None = None,
    ) -> None:
        """
        Upload raw bytes with explicit object name and metadata.
        """
        self.client.put_object(
            bucket_name=self.bucket_name,
            object_name=object_name,
            data=BytesIO(data),
            length=len(data),
            content_type=content_type,
            metadata=metadata,
        )

    # --------------------------------------------------

    def download_file(
        self,
        object_name: str,
        file_path: str,
    ) -> None:
        """
        Download file from MinIO to the given file_path.
        """

        self.client.fget_object(
            bucket_name=self.bucket_name,
            object_name=object_name,
            file_path=file_path,
        )

    # --------------------------------------------------

    def download_bytes(self, object_name: str) -> bytes:
        """
        Download object from MinIO as raw bytes (in-memory, no local file).
        """
        response = self.client.get_object(
            bucket_name=self.bucket_name,
            object_name=object_name,
        )
        try:
            return response.read()
        finally:
            response.close()
            response.release_conn()

    # --------------------------------------------------

    def delete_object(self, object_name: str) -> None:
        """
        Delete object from bucket.
        """

        self.client.remove_object(
            bucket_name=self.bucket_name,
            object_name=object_name,
        )

    # --------------------------------------------------

    def delete_objects_for_doc_ids(
        self,
        docs: list[DocumentIDTopicPair],
        prefix: str,
    ) -> DeleteObjectsByDocIDsResponse:
        """
        Delete all objects under:
            <prefix>/<doc_id>/*
        """

        if not docs:
            return {"deleted": [], "errors": []}

        delete_list = []

        # Collect objects
        for doc in docs:
            objects = self.client.list_objects(
                self.bucket_name,
                prefix=f"{prefix}/{doc.topic.value}/{doc.doc_id}",
                recursive=True,
            )

            delete_list.extend(DeleteObject(obj.object_name) for obj in objects)

        if not delete_list:
            return {"deleted": [], "errors": []}

        errors = []

        results = self.client.remove_objects(
            bucket_name=self.bucket_name,
            delete_object_list=delete_list,
        )

        for err in results:
            errors.append(
                {
                    "object_name": err.object_name,
                    "error": err.message,
                }
            )

        return DeleteObjectsByDocIDsResponse(
            deleted_count=len(delete_list),
            errors=errors,
        )

    # --------------------------------------------------

    def list_objects(self, prefix: str | None = None, include_metadata: bool = True) -> list[MinioStorageObject]:
        """List objects in bucket.

        Args:
            include_metadata: When False, skips per-object stat_object calls (faster).
                              Use False when only object names/sizes are needed.
        """
        objects = self.client.list_objects(
            self.bucket_name,
            prefix=prefix,
            recursive=True,
        )

        results = []
        for obj in objects:
            if include_metadata:
                stat_info = self.client.stat_object(self.bucket_name, obj.object_name)
                metadata = stat_info.metadata
            else:
                metadata = {}
            results.append(
                MinioStorageObject(
                    object_name=obj.object_name,
                    size=obj.size,
                    last_modified=obj.last_modified,
                    metadata=metadata,
                )
            )

        return results

    # --------------------------------------------------

    def get_metadata(self, object_name: str) -> dict:
        """Fetch metadata for an object."""
        stat_info = self.client.stat_object(self.bucket_name, object_name)
        return stat_info.metadata

    # --------------------------------------------------

    def presigned_get_url(self, object_name: str, expires_seconds: int = 3600) -> str:
        """
        Generate a presigned GET URL for an object.

        Uses the public-endpoint signer when configured so the URL resolves
        through the reverse proxy / public DNS instead of the internal endpoint.
        """
        return self.signer.presigned_get_object(
            bucket_name=self.bucket_name,
            object_name=object_name,
            expires=timedelta(seconds=expires_seconds),
        )
