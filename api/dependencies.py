# FastAPI dependency providers.
# Clients and services are created once at startup and stored on app.state (see main.py
# lifespan). These functions retrieve them per-request via `Depends(...)` — no
# re-initialization happens on each call, which keeps connection overhead low.
from fastapi import Request
from pymilvus import MilvusClient

from api.client.llm_client import LLMClient
from api.client.minio_client import MinioClient
from api.client.redis_client import RedisClient
from api.config import Settings
from api.services.corpus_storage import CorpusStorageService
from api.services.ingestion_service import IngestionService
from api.services.reranker import CrossEncoderReranker
from api.services.task_manager import TaskManager
from api.utils.markdown_parser import MarkdownParser


def get_minio_client(request: Request) -> MinioClient:
    return request.app.state.minio_client


def get_llm_client(request: Request) -> LLMClient:
    return request.app.state.llm_client


def get_milvus_client(request: Request) -> MilvusClient:
    return request.app.state.milvus_client


def get_redis_client(request: Request) -> RedisClient:
    return request.app.state.redis_client


def get_markdown_parser(request: Request) -> MarkdownParser:
    return request.app.state.markdown_parser


def get_task_manager(request: Request) -> TaskManager:
    return request.app.state.task_manager


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_corpus_storage(request: Request) -> CorpusStorageService:
    return request.app.state.corpus_storage


def get_ingestion_service(request: Request) -> IngestionService:
    return request.app.state.ingestion_service


def get_reranker(request: Request) -> CrossEncoderReranker:
    return request.app.state.reranker
