# Shared data models used across the API, services, and utilities.
# All Pydantic models — keeping them in one file avoids circular imports between
# routers and services, which would occur if each module defined its own schemas.
from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# -------------------------------
# Enums
# -------------------------------
"""
The below enums are available topics the user can select while ingesting or while trying to generate question-answer pairs.
"""


class TopicEnum(StrEnum):
    general = "general"  # generic fallback
    aviation = "aviation"
    finance = "finance"
    healthcare = "healthcare"
    technology = "technology"
    legal = "legal"
    hr = "hr"


class QuestionTypeEnum(StrEnum):
    SUBJECTIVE = "subjective"
    MCQ = "mcq"
    MATCH_MAKING = "match_making"
    ONE_WORD = "one_word"


class KnowledgeFeedModeEnum(StrEnum):
    TEXT = "text"  # LLM receives only chunk text
    IMAGE = "image"  # LLM receives only PDF page image
    HYBRID = "hybrid"  # LLM receives both chunk text and image (see inference-matching.md — image path not yet fully wired)


# -------------------------------
# Pymupdfllm parsing
# -------------------------------
class MarkdownParsedPage(BaseModel):
    metadata: dict[str, Any]
    text: str


class MarkdownParsedChunk(BaseModel):
    content: str  # The text of this chunk
    parent_headings: list[str] = Field(default_factory=list)  # Ordered list of ancestor headings
    page_num: int | None = None  # Page number from source document

    model_config = {"frozen": False}


class IngestedDocumentResponse(BaseModel):
    # All lists are parallel: index i across all fields refers to the same chunk.
    chunks: list[MarkdownParsedChunk]
    vectors: list[list[float]]  # dense embeddings — stored in Milvus FLOAT_VECTOR field
    doc_ids: list[str]  # same doc_id repeated for every chunk of the document
    file_names: list[str]  # original filenames (one per chunk, all identical within a doc)
    chunk_texts: list[str]  # raw chunk body — used for BM25 sparse_vector
    parent_headings: list[str]  # JSON-encoded list of ancestor heading strings
    headings_texts: list[str]  # space-joined heading path — used for BM25 headings_sparse_vector
    page_nums: list[int]  # source page numbers


# -------------------------------
# Collecting top similar Corpus chunks / Scoring mechanism
# -------------------------------
class GeneratedQA(BaseModel):
    question: str
    answer: str
    options: list[str] | None = None  # populated for MCQ type


class MatchSet(BaseModel):
    set_label: str
    pairs: list[dict[str, str]]
    answer: str


class LLMGeneratedResponse(BaseModel):
    questions: list[GeneratedQA] | None = None  # for subjective, mcq, one_word
    match_sets: list[MatchSet] | None = None  # for match_making
    raw: str | None = None  # fallback if JSON parsing fails


class InputQueryChunkMatch(BaseModel):
    corpus_chunk: str
    corpus_page_num: int | None = None
    score: float
    doc_url: str | None = None
    llm_generated_questions: LLMGeneratedResponse | None = None


class InputQueryDocumentMatchSummary(BaseModel):
    doc_id: str
    file_name: str
    doc_url: str
    compatibility_score: float


class InputQueryDocumentMatchReport(InputQueryDocumentMatchSummary):
    chunk_matches: list[InputQueryChunkMatch]


# -------------------------------
# Report Generation
# -------------------------------
class InputQueryOverallReportSummary(BaseModel):
    input_query: str
    total_matched_docs: int
    target_matching_files: list[InputQueryDocumentMatchSummary]


class InputQueryOverallReport(BaseModel):
    summary: InputQueryOverallReportSummary
    detailed_matches: list[InputQueryDocumentMatchReport]


# -------------------------------
# RAG Inference Pipeline
# -------------------------------
class RagInferencePipelineResponse(BaseModel):
    markdown_report_url: str
    results: list[InputQueryOverallReport]


# -------------------------------
# RAG Ingestion Pipeline Schemas
# -------------------------------
class RAGIngestionResponse(BaseModel):
    doc_ids: list[str]
    processed_count: int
    errors: list[str]


# -------------------------------
# LiteLLM Schemas
# -------------------------------
class ChatSource(BaseModel):
    doc_id: str
    file_name: str
    page_num: int | None = None
    doc_url: str
    relevance_score: float
    chunk_excerpt: str


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    sources: list[ChatSource] | None = None


# -------------------------------
# Milvus Schemas
# -------------------------------
class MilvusInsertResponse(BaseModel):
    insert_count: int
    delete_count: int
    upsert_count: int
    success_count: int
    error_count: int
    timestamp: int
    primary_keys: list[int]


class MilvusSearchHit(BaseModel):
    id: int
    distance: float
    doc_id: str | None
    file_name: str | None
    chunk_text: str | None
    parent_headings: list[str] | None
    page_num: int | None


# -------------------------------
# MinIO schemas
# -------------------------------
class TrainingPDFMetadata(BaseModel):
    original_filename: str = Field(alias="original-filename")


class ExtractedImageMetadata(BaseModel):
    page_num: int


MetadataSchema = TrainingPDFMetadata | ExtractedImageMetadata


class DeleteObjectsByDocIDsResponse(BaseModel):
    deleted_count: int
    errors: list[str]


class MinioStorageObject(BaseModel):
    object_name: str
    size: int
    last_modified: datetime
    metadata: dict[str, str]


# -------------------------------
# API Request/Response Models
# -------------------------------
class CorpusUploadResponse(BaseModel):
    doc_id: str
    filename: str


class ListCorpusDocuments(BaseModel):
    doc_id: str
    filename: str
    upload_date: str
    size: int
    topic: TopicEnum


class DocumentIDTopicPair(BaseModel):
    doc_id: str
    topic: TopicEnum


class ListExtractedDocumentImages(BaseModel):
    doc_id: str
    images: list[str]


class DocumentInferRoute(BaseModel):
    session_id: str
    result: RagInferencePipelineResponse


class PaginatedResponse(BaseModel):
    items: list[ListCorpusDocuments]
    total: int
    page: int
    limit: int
    has_next: bool


# -------------------------------
# Background Task Models
# -------------------------------
class TaskStatusEnum(StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskAcceptedResponse(BaseModel):
    task_id: str
    doc_ids: list[str]
    status: TaskStatusEnum
    poll_url: str


class TaskSummary(BaseModel):
    task_id: str
    status: TaskStatusEnum
    description: str | None = None
    progress: str | None = None
    elapsed_time: str | None = None
    error: str | None = None
    created_at: str | None = None
    updated_at: str | None = None


class TaskStatusResponse(TaskSummary):
    result: RAGIngestionResponse | None = None


class TaskListResponse(BaseModel):
    tasks: list[TaskStatusResponse]
    total: int
    not_found: list[str] = []


# -------------------------------
# Chatbot Response Models
# -------------------------------
class ChatResponse(BaseModel):
    answer: str
    sources: list[ChatSource]
    session_id: str | None = None
    history: list[ChatMessage] = []


class SessionSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    topic: str
    created_at: str
    last_updated: str
    message_count: int
    preview: str


class UserSessionListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sessions: list[SessionSummary]
    total: int


class SessionDetailResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    topic: str
    created_at: str
    last_updated: str
    message_count: int
    preview: str
    history: list[ChatMessage]
