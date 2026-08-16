# Application-wide settings loaded from environment variables (or .env file).
# Each nested class maps to an env-var group via the __ delimiter:
#   LLM__PROVIDER=openai  →  settings.llm.provider
# Never import `settings` inside a class body or module-level expression that runs at
# import time — use FastAPI dependency injection (get_settings) for testability.
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


# Security Configuration
class SecuritySettings(BaseSettings):
    # CORS origins allowed to call this API.
    # Keep as ["*"] only for local dev — set explicit origins before deploying.
    cors_allowed_origins: list[str] = ["*"]
    # When False, the global exception handler returns a generic message instead
    # of exposing internal error details to callers. Always False in production.
    expose_exception_details: bool = True


# LLM Configuration
class LLMSettings(BaseSettings):
    provider: Literal["ollama", "openai", "gemini"] = "ollama"
    ollama_chat_model: str = "gemma3:4b"
    ollama_embed_model: str = "nomic-embed-text"
    ollama_api_base: str = "http://localhost:11434"
    litellm_chat_model: str = "ollama/gemma3:4b"
    litellm_embed_model: str = "ollama/nomic-embed-text"
    chat_temperature: float = 0.2


# Milvus Configuration
class MilvusSettings(BaseSettings):
    host: str = "localhost"
    port: int = 19530
    corpus_collection_name: str = "rigil_corpus"
    input_collection_name: str = "rigil_input"
    vector_dim: int = 768
    metric_type: Literal["COSINE", "EUCLIDEAN", "IP"] = "COSINE"
    index_type: str = "HNSW"
    M: int = 16
    ef_construction: int = 200
    # Hybrid-search fusion weights for Milvus WeightedRanker.
    # Must be in [0, 1]. Heading BM25 is weighted higher so that queries
    # hitting a chunk's heading path outrank body-only matches.
    dense_weight: float = 0.25
    chunk_bm25_weight: float = 0.25
    heading_bm25_weight: float = 0.5
    # Search tuning
    hnsw_ef: int = 64  # HNSW search-time ef parameter
    search_over_fetch_factor: int = 4  # over-fetch multiplier before fusion re-ranking


# Chat / Infer Defaults
class ChatSettings(BaseSettings):
    # Default limits used by the infer endpoint query parameters
    default_infer_limit: int = 5
    default_infer_group_size: int = 1
    default_questions_per_chunk: int = 2
    # Chat endpoint search limits
    chat_search_limit: int = 5
    chat_search_group_size: int = 2
    # Truncation for chunk excerpts shown in chat source citations
    chunk_excerpt_chars: int = 200
    # How many recent history turns to pass to the LLM as context (vs. the full history limit in RedisSettings)
    history_context_k: int = 5
    # Max files per upload request
    max_files_per_upload: int = 20
    # Allowed document file extensions
    allowed_extensions: list[str] = [".pdf", ".docx", ".txt", ".md"]


# MinIO Configuration
class MinioSettings(BaseSettings):
    endpoint: str = "localhost:9000"
    access_key: str = "minioadmin"
    secret_key: str = "minioadmin"  # noqa: S105
    bucket_name: str = "rag-storage"
    secure: bool = False
    # Public endpoint used only when signing presigned GET URLs. When set, the
    # SDK signs the URL against this host so browsers can resolve it through the
    # reverse proxy / public DNS instead of the internal MINIO__ENDPOINT IP.
    # Leave None to fall back to `endpoint` (single-host / dev setups).
    public_endpoint: str | None = None
    public_secure: bool = True
    region: str | None = None


# Redis Configuration
class RedisSettings(BaseSettings):
    redis_url: str = "redis://localhost:6379"
    chat_history_limit: int = 20  # max turns returned in response history
    session_preview_chars: int = 120


# Background Task Configuration
class BackgroundTaskSettings(BaseSettings):
    use_background_tasks: bool = True
    task_result_ttl_seconds: int = 3600  # 1 hour


# Document Parser Configuration
class DocParserSettings(BaseSettings):
    include_header_in_chunk_content: bool = True
    chunk_max_tokens: int = 500
    chunk_overlap_tokens: int = 100
    # cl100k_base matches the tokenizer used by OpenAI text-embedding-3 and GPT-4.
    # Switch to r50k_base for older models like text-davinci-003.
    tiktoken_encoder: str = "cl100k_base"
    # When True, uses the hybrid pymupdf4llm + docling pipeline for table/image extraction.
    # When False, falls back to text-only parsing (faster but misses tables and images).
    use_docling_hybrid: bool = True


class Settings(BaseSettings):
    app_env: Literal["dev", "prod"] = "prod"
    app_port: int = 8001
    security: SecuritySettings = SecuritySettings()
    llm: LLMSettings = LLMSettings()
    milvus: MilvusSettings = MilvusSettings()
    minio: MinioSettings = MinioSettings()
    redis: RedisSettings = RedisSettings()
    doc_parser: DocParserSettings = DocParserSettings()
    background: BackgroundTaskSettings = BackgroundTaskSettings()
    chat: ChatSettings = ChatSettings()

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",  # maps LLM__PROVIDER → llm.provider
        extra="ignore",
    )


# Create instance to import `settings` anywhere in your FastAPI app
settings = Settings()
