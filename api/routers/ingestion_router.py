# Document ingestion endpoints: parse, embed, and index corpus documents into Milvus.
# Three entry points are provided:
#   1. ingest-doc-ids  — index documents already uploaded to MinIO (by doc_id)
#   2. ingest-file     — single-file upload + ingest (convenient for Swagger testing)
#   3. ingest-files    — batch upload + ingest
# All three delegate to IngestionService.start(), which routes to sync or background
# processing based on the BACKGROUND__USE_BACKGROUND_TASKS config flag.
import logging

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, UploadFile

from api.dependencies import get_ingestion_service
from api.exceptions import ValidationError
from api.models import DocumentIDTopicPair, TopicEnum
from api.services.ingestion_service import IngestionService
from api.utils.upload_utils import validate_uploads

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/corpus", tags=["Ingestion"])


@router.post("/document/ingest-doc-ids")
def ingest_corpus_documents(
    docs: list[DocumentIDTopicPair],
    background: BackgroundTasks,
    ingestion_service: IngestionService = Depends(get_ingestion_service),  # noqa: B008
):
    """Process and index a list of document IDs into the vector database."""
    logger.info("ingest_corpus_documents: started for %d doc(s)", len(docs))
    doc_ids = [doc.doc_id for doc in docs]
    topics = [doc.topic for doc in docs]
    return ingestion_service.start(
        doc_ids=doc_ids,
        topics=topics,
        description=f"Ingest {len(doc_ids)} document(s) by ID into the vector database",
        background=background,
    )


# Kept for single-file ingestion via Swagger UI.
@router.post("/document/ingest-file")
async def upload_and_ingest_file(
    background: BackgroundTasks,
    file: UploadFile = File(...),  # noqa: B008
    topic: TopicEnum = Form(...),  # noqa: B008
    ingestion_service: IngestionService = Depends(get_ingestion_service),  # noqa: B008
):
    """Upload a file to MinIO and ingest it into Milvus."""
    validate_uploads([file])
    logger.info(
        "upload_and_ingest_file: started for file='%s', topic=%s",
        file.filename,
        topic.value,
    )
    doc_ids = await ingestion_service.upload_files([file], topic)
    return ingestion_service.start(
        doc_ids=doc_ids,
        topics=[topic],
        description=f"Upload and ingest '{file.filename}' ({topic.value})",
        background=background,
    )


@router.post("/document/ingest-files")
async def upload_and_ingest_files(
    background: BackgroundTasks,
    topic: TopicEnum,
    files: list[UploadFile] = File(...),  # noqa: B008
    ingestion_service: IngestionService = Depends(get_ingestion_service),  # noqa: B008
):
    """Upload multiple files to MinIO and ingest them."""
    if not files:
        raise ValidationError("No files uploaded.")
    validate_uploads(files)
    logger.info(
        "upload_and_ingest_files: started for %d file(s), topic=%s",
        len(files),
        topic.value,
    )
    doc_ids = await ingestion_service.upload_files(files, topic)
    filenames = ", ".join(f.filename for f in files)
    return ingestion_service.start(
        doc_ids=doc_ids,
        topics=[topic] * len(doc_ids),
        description=f"Upload and ingest {len(files)} file(s) ({topic.value}): {filenames}",
        background=background,
    )
