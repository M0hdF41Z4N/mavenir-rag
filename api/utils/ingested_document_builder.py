# Converts parsed document chunks into the parallel-list format expected by MilvusClient.insert().
# Handles per-chunk embedding calls and skips chunks whose embedding fails so one bad
# chunk doesn't abort the entire document.
import json
import logging

from api.client.llm_client import LLMClient
from api.models import IngestedDocumentResponse, MarkdownParsedChunk

logger = logging.getLogger(__name__)


def build_ingested_document(
    chunks: list[MarkdownParsedChunk],
    llm_client: LLMClient,
    doc_id: str,
    original_filename: str,
    log_prefix: str = "",
) -> IngestedDocumentResponse:
    """Build IngestedDocumentResponse from chunks with embeddings."""
    result_chunks = []
    vectors = []
    doc_ids = []
    file_names = []
    chunk_texts = []
    parent_headings = []
    headings_texts = []
    page_nums = []

    for chunk in chunks:
        try:
            heading_prefix = " > ".join(chunk.parent_headings)
            # Prepend the heading breadcrumb to the chunk content before embedding so the
            # dense vector captures structural context (e.g. "Section 3 > Safety Procedures")
            # rather than floating body text alone. The heading_prefix is stored separately
            # as headings_text for the BM25 sparse index.
            embed_text = f"{heading_prefix}\n\n{chunk.content}" if heading_prefix else chunk.content
            embedding = llm_client.embed(embed_text)

            result_chunks.append(chunk)
            vectors.append(embedding)
            doc_ids.append(doc_id)
            file_names.append(original_filename)
            chunk_texts.append(chunk.content)
            parent_headings.append(json.dumps(chunk.parent_headings))
            headings_texts.append(heading_prefix)
            page_nums.append(chunk.page_num)
        except Exception as err:
            logger.warning(
                "%sEmbedding failed for %s (doc_id=%s): %s",
                log_prefix,
                original_filename,
                doc_id,
                err,
            )

    return IngestedDocumentResponse(
        chunks=result_chunks,
        vectors=vectors,
        doc_ids=doc_ids,
        file_names=file_names,
        chunk_texts=chunk_texts,
        parent_headings=parent_headings,
        headings_texts=headings_texts,
        page_nums=page_nums,
    )
