# Corpus management endpoints: upload, list, download, and delete corpus documents.
# "Corpus" here means the reference document set that gets ingested into Milvus and
# queried during inference. Raw files live in MinIO; their vector embeddings live in Milvus.
# MinIO object path convention:  corpus/<topic>/<doc_id>/<ext>
import logging
import os
import tempfile
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Query, UploadFile
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

from api.client.milvus_client import MilvusClient
from api.client.minio_client import MinioClient
from api.dependencies import get_milvus_client, get_minio_client
from api.exceptions import NotFoundError, ValidationError
from api.models import (
    CorpusUploadResponse,
    DocumentIDTopicPair,
    ListCorpusDocuments,
    ListExtractedDocumentImages,
    PaginatedResponse,
    TopicEnum,
)
from api.utils.timing import timed
from api.utils.upload_utils import validate_uploads

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/corpus", tags=["Corpus"])


def _minio_object_to_file_response(minio_client: MinioClient, object_name: str, filename: str) -> FileResponse:
    """Download a MinIO object to a temp file and return it as a FileResponse.
    The temp file is deleted automatically after the response streams.
    """
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp_path = tmp.name
    minio_client.download_file(object_name, tmp_path)
    return FileResponse(
        tmp_path,
        filename=filename,
        background=BackgroundTask(os.unlink, tmp_path),
    )


@router.post("/document/upload", response_model=list[CorpusUploadResponse])
@timed("upload_corpus_documents", logger)
async def upload_corpus_documents(
    topic: TopicEnum,
    files: list[UploadFile] = File(...),  # noqa: B008
    minio_client: MinioClient = Depends(get_minio_client),  # noqa: B008
) -> list[CorpusUploadResponse]:
    """Upload multiple documents to the corpus directory."""
    validate_uploads(files)
    logger.info("Uploading %d file(s) for topic=%s", len(files), topic.value)
    responses = []

    for file in files:
        doc_id = str(uuid.uuid4())
        ext = Path(file.filename).suffix
        object_name = f"corpus/{topic.value}/{doc_id}/{ext}"

        # Save to temp file to upload to MinIO
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        try:
            minio_client.upload_file(
                file_path=tmp_path,
                object_name=object_name,
                metadata={"original-filename": file.filename},
            )
            responses.append(CorpusUploadResponse(doc_id=doc_id, filename=file.filename))
        finally:
            os.unlink(tmp_path)

    return responses


@router.get("/document/list-corpus", response_model=PaginatedResponse)
@timed("list_corpus_documents", logger)
def list_corpus_documents(
    page: int = Query(1, ge=1),  # noqa: B008
    limit: int = Query(10, ge=1, le=100),  # noqa: B008
    sort_order: int = Query(0, ge=0, le=1, description="0 = desc (latest first), 1 = asc (oldest first)"),  # noqa: B008
    search: str | None = Query(None),  # noqa: B008
    query_topic: TopicEnum | None = Query(None),  # noqa: B008
    minio_client: MinioClient = Depends(get_minio_client),  # noqa: B008
) -> PaginatedResponse:
    """List all documents in the corpus."""
    objects = list(minio_client.list_objects(prefix="corpus/"))
    logger.info("Found %d objects in MinIO", len(objects))
    results: list[ListCorpusDocuments] = []

    for obj in objects:
        # Extract doc_id from path: corpus/uuid.pdf
        parts = obj.object_name.split("/")
        # Expect parts = "corpus/{topic}/{doc_id}"
        if len(parts) < 3:
            continue
        doc_topic = parts[1]  # extract topic
        doc_id = parts[-2]
        filename = obj.metadata.get("x-amz-meta-original-filename", doc_id)
        results.append(
            ListCorpusDocuments(
                doc_id=doc_id,
                filename=filename,
                topic=TopicEnum(doc_topic),
                upload_date=str(obj.last_modified),
                size=obj.size,
            )
        )

    # STEP 1: Filter by topic
    if query_topic:
        results = [doc for doc in results if doc.topic == query_topic]

    # STEP 2: Search by filename
    if search:
        if not search.isalnum():
            raise ValidationError("Search must be alphanumeric only")
        search_lower = search.lower()
        results = [doc for doc in results if search_lower in doc.filename.lower() or search_lower in doc.topic.value.lower()]

    # STEP 3: Sort by upload_date
    reverse = sort_order == 0
    results.sort(key=lambda x: x.upload_date, reverse=reverse)
    # Convert datetime to string AFTER sorting — string comparison on dates breaks order
    for doc in results:
        doc.upload_date = str(doc.upload_date)

    total = len(results)
    start = (page - 1) * limit
    end = start + limit

    if start >= total and total != 0:
        raise ValidationError(f"Page {page} is out of range. Total documents: {total}")

    return PaginatedResponse(
        items=results[start:end],
        total=total,
        page=page,
        limit=limit,
        has_next=end < total,
    )


@router.get("/document/{doc_id}/download", response_class=FileResponse)
@timed("download_corpus_document", logger)
def download_corpus_document(
    doc_id: str,
    topic: TopicEnum = Query(...),  # noqa: B008
    minio_client: MinioClient = Depends(get_minio_client),  # noqa: B008
) -> FileResponse:
    """Download a corpus document by its ID."""
    # Find the object with this prefix
    objects = minio_client.list_objects(prefix=f"corpus/{topic.value}/{doc_id}")
    if not objects:
        raise NotFoundError("Document not found")

    obj = objects[0]
    filename = obj.metadata.get("x-amz-meta-original-filename", Path(obj.object_name).name)
    return _minio_object_to_file_response(minio_client, obj.object_name, filename)


@router.delete("/document/delete")
@timed("delete_corpus_documents", logger)
def delete_corpus_documents(
    docs: list[DocumentIDTopicPair],
    minio_client: MinioClient = Depends(get_minio_client),  # noqa: B008
    milvus_client: MilvusClient = Depends(get_milvus_client),  # noqa: B008
):
    """Delete corpus documents and their embeddings."""
    if not docs:
        raise ValidationError("doc_ids cannot be empty")

    # Delete from MinIO (BULK)
    minio_result = minio_client.delete_objects_for_doc_ids(docs=docs, prefix="corpus")
    # Delete embeddings from Milvus (BATCH)
    milvus_result = milvus_client.delete_embeddings_for_doc_ids(docs=docs)

    return {
        "requested_docs": docs,
        "minio_deleted_objects": minio_result,
        "milvus_delete_result": str(milvus_result),
    }


@router.get("/document/{doc_id}/images", response_model=ListExtractedDocumentImages)
@timed("list_doc_images", logger)
def list_doc_images(
    doc_id: str,
    topic: TopicEnum,
    minio_client: MinioClient = Depends(get_minio_client),  # noqa: B008
) -> ListExtractedDocumentImages:
    """Return a list of all images associated with a document ID."""
    # Use prefix for images
    prefix = f"parsed_images/{topic.value}/{doc_id}/"
    objects = minio_client.client.list_objects(minio_client.bucket_name, prefix=prefix, recursive=True)
    image_list = [obj.object_name for obj in objects]

    if not image_list:
        raise NotFoundError("No images found for this doc_id")

    return ListExtractedDocumentImages(doc_id=doc_id, images=image_list)


@router.get("/document/{doc_id}/images/download", response_class=FileResponse)
@timed("download_corpus_image", logger)
def download_corpus_image(
    doc_id: str,
    topic: TopicEnum,
    filename: str | None = None,
    minio_client: MinioClient = Depends(get_minio_client),  # noqa: B008
) -> FileResponse:
    """Download an image associated with a document ID."""
    # Trailing slash required — without it the prefix matches other doc_ids that share the same prefix
    # Use correct prefix with trailing slash
    objects = minio_client.list_objects(prefix=f"parsed_images/{topic.value}/{doc_id}/")
    if not objects:
        raise NotFoundError("Document not found")

    # If filename provided, find that object
    if filename:
        obj = next((o for o in objects if Path(o.object_name).name == filename), None)
        if not obj:
            raise NotFoundError("File not found")
    else:
        obj = objects[0]  # default to first image

    dl_filename = obj.metadata.get("x-amz-meta-original-filename", Path(obj.object_name).name)
    return _minio_object_to_file_response(minio_client, obj.object_name, dl_filename)
