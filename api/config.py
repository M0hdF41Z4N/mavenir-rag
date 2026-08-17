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
    # Model used by the post-generation groundedness verifier. When None, the verifier
    # reuses litellm_chat_model — but a generator grading its own model family is a weak
    # check, so point this at a different (ideally stronger) model on the SAME provider,
    # e.g. "ollama/llama3.1:8b". Cross-provider verifiers (different api_base/keys) are
    # not supported here — the verifier shares this client's api_base.
    litellm_verifier_model: str | None = None
    # Temperature for the creative question-generation path (subjective/MCQ/etc.).
    chat_temperature: float = 0.2
    # Temperature for the corpus-grounded chat path. Pinned to 0.0 so grounded
    # answers are deterministic and the model has no sampling room to drift off
    # the retrieved evidence — a lower-hallucination default than chat_temperature.
    grounded_temperature: float = 0.0


# Milvus Configuration
class MilvusSettings(BaseSettings):
    host: str = "localhost"
    port: int = 19530
    corpus_collection_name: str = "mavenir_rag_corpus"
    input_collection_name: str = "mavenir_rag_input"
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
    # How a document's compatibility score is aggregated from its chunk scores.
    #   "max"         — the single best chunk represents the doc (default; a strong hit
    #                   isn't diluted by unrelated chunks from the same doc).
    #   "mean"        — arithmetic mean of all chunk scores (legacy behavior).
    #   "top_k_mean"  — mean of the top `doc_score_top_k` chunk scores (a middle ground).
    # "max" is the more defensible basis for ranking/thresholds than "mean", which lets a
    # pile of weak chunks drag down a doc that has one excellent match.
    doc_score_aggregation: Literal["max", "mean", "top_k_mean"] = "max"
    doc_score_top_k: int = 3  # used only when doc_score_aggregation == "top_k_mean"
    # Evidence-sufficiency gate: minimum fused hybrid-search score the top chunk
    # must reach for the retrieved context to be considered sufficient to answer.
    # When the best hit falls below this floor (or nothing is retrieved), the chat
    # endpoint abstains deterministically instead of asking the LLM to answer from
    # weak/irrelevant evidence — the cheapest, un-foolable hallucination control.
    # NOTE: hybrid search fuses dense + BM25 scores via WeightedRanker, so this is
    # a fused weighted-sum score, not a raw cosine similarity. Tune per corpus.
    min_evidence_score: float = 0.0
    # Post-generation groundedness verifier: after the answer is generated, a second
    # LLM pass decomposes it into claims and checks each against the retrieved chunks.
    # Off by default so behavior is unchanged until deliberately enabled (it costs one
    # extra LLM call per turn). A separate pass judging the answer is a much stronger
    # control than trusting the generator to police itself.
    enable_groundedness_check: bool = False
    # When the verifier finds unsupported claims and this is True, the endpoint replaces
    # the answer with the canonical abstention instead of returning a partially
    # hallucinated answer. When False, the answer is returned as-is but the
    # GroundednessReport still flags is_grounded=False so callers/UI can react.
    abstain_on_ungrounded: bool = True
    # Contradiction check: a separate LLM pass scans the retrieved chunks for conflicting
    # statements (e.g. two docs naming different CEOs) and reports the conflicting pairs.
    # Silently picking whichever chunk ranked higher is a hallucination risk; surfacing the
    # disagreement is safer. Off by default (one extra LLM call per turn).
    enable_contradiction_check: bool = False
    # When contradictions are found and this is True, prepend a short disclosure to the
    # answer so the user knows the sources disagree. The ContradictionReport is attached
    # to the response either way.
    disclose_contradictions: bool = True
    # When the groundedness check runs and passes, append a per-claim citation breakdown
    # ([n] markers per claim) to the answer text. The structured citations are always put
    # on the response's GroundednessReport regardless; this flag only controls whether the
    # human-readable breakdown is spliced into `answer`. Off by default to keep answer text
    # unchanged for existing clients.
    render_claim_citations: bool = False
    # Truncation for chunk excerpts shown in chat source citations
    chunk_excerpt_chars: int = 200
    # How many recent history turns to pass to the LLM as context (vs. the full history limit in RedisSettings)
    history_context_k: int = 5
    # Max files per upload request
    max_files_per_upload: int = 20
    # Allowed document file extensions
    allowed_extensions: list[str] = [".pdf", ".docx", ".txt", ".md"]


# Reranker Configuration
class RerankSettings(BaseSettings):
    # Cross-encoder reranking sits between hybrid retrieval and the LLM: over-fetch a
    # wide candidate pool, score each (query, chunk) pair with a cross-encoder, keep the
    # top-k. Milvus fusion (WeightedRanker) is not reranking — a cross-encoder that reads
    # query and chunk together is the highest-leverage retrieval-quality improvement.
    # Off by default: enabling it pulls in sentence-transformers (torch) and loads a model.
    enable_rerank: bool = False
    # A sentence-transformers CrossEncoder model id.
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    # How many top candidates to keep after reranking (the k in over-fetch → rerank → k).
    rerank_top_k: int = 5
    # Multiplier applied to the caller's requested limit to size the candidate pool that
    # is reranked. e.g. want 5 back, fetch 5 * 4 = 20, rerank, keep 5.
    rerank_over_fetch_factor: int = 4


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
    rerank: RerankSettings = RerankSettings()
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
