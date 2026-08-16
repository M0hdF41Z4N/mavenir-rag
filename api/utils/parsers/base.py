# Base parser protocol enables pluggable document parsing strategies, allowing domain-specific
# extraction logic (topic-aware, format-specific) without coupling to ingestion service.
# Provides extensibility for future RAG enhancements: custom chunk strategies, format adapters,
# and LLM-driven preprocessing tailored to different document types.
from typing import Protocol

from api.client.llm_client import LLMClient
from api.client.minio_client import MinioClient
from api.models import IngestedDocumentResponse, TopicEnum
from api.utils.markdown_parser import MarkdownParser


class DocumentParser(Protocol):
    """Callable protocol shared by all document parser implementations."""

    def __call__(
        self,
        doc_id: str,
        topic: TopicEnum,
        file_name: str,
        minio_client: MinioClient,
        llm_client: LLMClient,
        markdown_parser: MarkdownParser,
    ) -> IngestedDocumentResponse: ...
