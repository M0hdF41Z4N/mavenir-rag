# The two core RAG pipelines:
#
#   rag_ingestion_pipeline  — offline: for each doc_id, parse → embed → insert into Milvus.
#                             Called by IngestionService (sync or as a background task).
#
#   rag_inference_pipeline  — online: embed input queries → hybrid Milvus search → score docs
#                             → generate Q&A per chunk via LLM → write Markdown report to MinIO.
#                             Called synchronously from the /document/infer endpoint.
import logging
from collections.abc import Callable
from dataclasses import dataclass

from pymilvus import MilvusClient

from api.client.llm_client import LLMClient
from api.client.minio_client import MinioClient
from api.client.redis_client import RedisClient
from api.config import settings
from api.models import (
    KnowledgeFeedModeEnum,
    QuestionTypeEnum,
    RagInferencePipelineResponse,
    RAGIngestionResponse,
    TopicEnum,
)
from api.services.corpus_storage import CorpusStorageService
from api.services.question_generator import QuestionGenerationParams, generate_questions_from_chunks
from api.utils.markdown_parser import MarkdownParser
from api.utils.parsers import get_parser
from api.utils.report_generator import generate_markdown_report
from api.utils.scoring import generate_doc_match_report

logger = logging.getLogger(__name__)


@dataclass
class RagInferenceConfig:
    username: str
    session_id: str
    input_queries: list[str]
    topic: TopicEnum
    limit: int
    group_size: int
    questions_per_top_chunks: int
    question_answer_type: QuestionTypeEnum
    knowledge_feed_mode: KnowledgeFeedModeEnum


@dataclass
class RagClients:
    minio_client: MinioClient
    milvus_client: MilvusClient
    llm_client: LLMClient
    redis_client: RedisClient
    corpus_storage: CorpusStorageService


def rag_ingestion_pipeline(
    topics: list[TopicEnum],
    doc_ids: list[str],
    minio_client: MinioClient,
    milvus_client: MilvusClient,
    llm_client: LLMClient,
    markdown_parser: MarkdownParser,
    corpus_storage: CorpusStorageService,
    progress_callback: Callable[[str], None] | None = None,
) -> RAGIngestionResponse:
    total_processed = 0
    errors = []

    for i, (doc_id, topic) in enumerate(zip(doc_ids, topics, strict=True)):
        if progress_callback:
            progress_callback(f"{i}/{len(doc_ids)} documents processed")
        try:
            # Find object path
            object_name = corpus_storage.object_name_for(doc_id, topic)
            if not object_name:
                errors.append(f"Document {doc_id} not found in MinIO")
                continue

            result = get_parser(settings)(
                doc_id=doc_id,
                topic=topic,
                file_name=object_name,
                minio_client=minio_client,
                llm_client=llm_client,
                markdown_parser=markdown_parser,
            )

            # Insert into Milvus
            milvus_client.insert(
                vectors=result.vectors,
                doc_ids=result.doc_ids,
                topics=[topic] * len(result.vectors),
                file_names=result.file_names,
                chunk_texts=result.chunk_texts,
                parent_headings=result.parent_headings,
                headings_texts=result.headings_texts,
                page_nums=result.page_nums,
            )
            total_processed += 1

        except Exception as e:
            errors.append(f"Error processing {doc_id}: {str(e)}")

    return RAGIngestionResponse(
        doc_ids=doc_ids,
        processed_count=total_processed,
        errors=errors,
    )


def rag_inference_pipeline(
    config: RagInferenceConfig,
    clients: RagClients,
) -> RagInferencePipelineResponse:
    # Generate embeddings for each input query
    query_vectors = [clients.llm_client.embed(query) for query in config.input_queries]

    # Search Milvus (hybrid: dense vectors + BM25 text)
    milvus_search_results = clients.milvus_client.search(
        query_vectors=query_vectors,
        query_texts=config.input_queries,
        limit=config.limit,
        group_size=config.group_size,
        expr=f'topic == "{config.topic.value}"',
    )

    # Generate scores for each corpus doc obtained from Milvus
    milvus_reports = generate_doc_match_report(
        topic=config.topic,
        milvus_search_results=milvus_search_results,
        input_queries=config.input_queries,
        corpus_storage=clients.corpus_storage,
        aggregation=settings.chat.doc_score_aggregation,
        top_k=settings.chat.doc_score_top_k,
    )

    llm_enriched_report = generate_questions_from_chunks(
        milvus_reports=milvus_reports,
        params=QuestionGenerationParams(
            username=config.username,
            session_id=config.session_id,
            question_answer_type=config.question_answer_type,
            knowledge_feed_mode=config.knowledge_feed_mode,
            total_questions=config.questions_per_top_chunks,
            llm_client=clients.llm_client,
            minio_client=clients.minio_client,
            redis_client=clients.redis_client,
        ),
    )

    markdown_report_url = generate_markdown_report(llm_enriched_report, "sample_report.md", minio_client=clients.minio_client)

    return RagInferencePipelineResponse(markdown_report_url=markdown_report_url, results=llm_enriched_report)
