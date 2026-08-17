# Inference and chat endpoints.
#   /auth/token      — exchange a Keycloak refresh token for a short-lived access token
#   /document/infer  — full RAG pipeline: embed input queries → hybrid search Milvus →
#                      score matched corpus docs → generate Q&A pairs per chunk → write report
#   /document/chat   — stateful RAG chatbot: retrieve context → build LLM messages with
#                      Redis chat history → answer grounded strictly in corpus
#   /document/session (DELETE) — clear a Redis chat session
# All /document/* endpoints require a valid Keycloak Bearer token.
import logging
import uuid

from fastapi import APIRouter, Depends, Query

from api.client.llm_client import LLMClient
from api.client.milvus_client import MilvusClient
from api.client.minio_client import MinioClient
from api.client.redis_client import RedisClient
from api.config import settings
from api.dependencies import (
    get_corpus_storage,
    get_llm_client,
    get_milvus_client,
    get_minio_client,
    get_redis_client,
    get_reranker,
)
from api.exceptions import NotFoundError
from api.models import (
    ChatMessage,
    ChatResponse,
    ChatSource,
    ContradictionReport,
    DocumentInferRoute,
    GroundednessReport,
    KnowledgeFeedModeEnum,
    MilvusSearchHit,
    QuestionTypeEnum,
    RagInferencePipelineResponse,
    SessionDetailResponse,
    SessionSummary,
    TopicEnum,
    UserSessionListResponse,
)
from api.services.contradiction_detector import (
    detect_contradictions,
    render_contradiction_notice,
)
from api.services.corpus_storage import CorpusStorageService
from api.services.groundedness_verifier import (
    render_cited_answer,
    resolve_citations,
    verify_groundedness,
)
from api.services.reranker import CrossEncoderReranker
from api.utils.keycloak_helper import (
    AuthenticatedKeycloakUser,
    KeycloakTokenRequest,
    KeycloakTokenResponse,
    create_access_token,
    require_keycloak_user,
)
from api.utils.llm_prompts import CHATBOT_SYSTEM_PROMPT, NO_INFO_ANSWER, NO_INFO_DETECTOR
from api.utils.rag_pipeline import RagClients, RagInferenceConfig, rag_inference_pipeline

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/mavenir-rag", tags=["Matcher"])


@router.post("/auth/token", response_model=KeycloakTokenResponse)
async def create_keycloak_token(payload: KeycloakTokenRequest) -> KeycloakTokenResponse:
    """Exchange a Keycloak refresh token for a short-lived access token."""
    logger.info(
        "auth/token called. requested_user_id=%s requested_username=%s",
        payload.user_id,
        payload.username,
    )
    return await create_access_token(payload)


@router.post("/document/infer", response_model=DocumentInferRoute)
async def run_similarity_search(
    topic: TopicEnum = Query(..., description="Filter docs by topic"),  # noqa: B008
    question_answer_type: QuestionTypeEnum = Query(  # noqa: B008
        QuestionTypeEnum.SUBJECTIVE,
        description="Type of question-answer generation: subjective, one_word, mcq, match_making",
    ),
    knowledge_feed_mode: KnowledgeFeedModeEnum = Query(  # noqa: B008
        KnowledgeFeedModeEnum.TEXT,
        description="How to feed knowledge to LLM: 'text' (chunk only), 'image' (PDF page image only) or 'hybrid' (chunk + PDF page image)",
    ),
    input_queries: list[str] = Query(  # noqa: B008
        ...,
        min_length=1,
        description="List of document IDs to use as input for similarity search",
    ),
    limit: int = Query(  # noqa: B008
        settings.chat.default_infer_limit,
        description="Retrieve the top chunk of only the top `limit` files from the corpus after similarity search is done for the given input chunk.",
    ),
    group_size: int = Query(  # noqa: B008
        settings.chat.default_infer_group_size,
        description="Max top chunks we want Milvus to consider based upon which it gives aggregated score and returns the top `limit` docs along with their `group_size` chunks.",
    ),
    questions_per_top_chunks: int = Query(  # noqa: B008
        settings.chat.default_questions_per_chunk,
        description="Total questions for the LLM to generate for each chunk.",
    ),
    session_id: str | None = Query(default=None, description="Optional session ID to maintain conversation context"),  # noqa: B008
    current_user: AuthenticatedKeycloakUser = Depends(require_keycloak_user),  # noqa: B008
    llm_client: LLMClient = Depends(get_llm_client),  # noqa: B008
    redis_client: RedisClient = Depends(get_redis_client),  # noqa: B008
    milvus_client: MilvusClient = Depends(get_milvus_client),  # noqa: B008
    minio_client: MinioClient = Depends(get_minio_client),  # noqa: B008
    corpus_storage: CorpusStorageService = Depends(get_corpus_storage),  # noqa: B008
) -> DocumentInferRoute:
    is_new = session_id is None
    if session_id is None:
        session_id = str(uuid.uuid4())

    logger.info(
        "document/infer called by user_id=%s username=%s topic=%s queries=%d session_id=%s",
        current_user.user_id,
        current_user.username,
        topic,
        len(input_queries),
        session_id,
    )

    result: RagInferencePipelineResponse = rag_inference_pipeline(
        config=RagInferenceConfig(
            username=current_user.username,
            session_id=session_id,
            input_queries=input_queries,
            topic=topic,
            limit=limit,
            group_size=group_size,
            questions_per_top_chunks=questions_per_top_chunks,
            question_answer_type=question_answer_type,
            knowledge_feed_mode=knowledge_feed_mode,
        ),
        clients=RagClients(
            minio_client=minio_client,
            milvus_client=milvus_client,
            llm_client=llm_client,
            redis_client=redis_client,
            corpus_storage=corpus_storage,
        ),
    )

    redis_client.register_session(
        username=current_user.username,
        session_id=session_id,
        topic=topic.value,
        first_query=input_queries[0],
        is_new=is_new,
    )

    return {"session_id": session_id, "result": result}


def _search_corpus(
    query: str,
    topic: TopicEnum,
    llm_client: LLMClient,
    milvus_client: MilvusClient,
    reranker: CrossEncoderReranker,
) -> list[MilvusSearchHit]:
    """Retrieve corpus chunks for the query, optionally cross-encoder reranked.

    When reranking is enabled, over-fetch a wider candidate pool from Milvus, re-score it
    with the cross-encoder, and keep the top-k — putting genuinely relevant chunks in front
    of the LLM. When disabled, behaves exactly as before (fusion-ranked, `chat_search_limit`).
    """
    query_vector = llm_client.embed(query)
    limit = settings.chat.chat_search_limit
    if settings.rerank.enable_rerank:
        limit = settings.chat.chat_search_limit * settings.rerank.rerank_over_fetch_factor

    results = milvus_client.search(
        query_vectors=[query_vector],
        query_texts=[query],
        limit=limit,
        group_size=settings.chat.chat_search_group_size,
        expr=f'topic == "{topic.value}"',
    )
    hits = results[0]
    if settings.rerank.enable_rerank:
        hits = reranker.rerank(query, hits, top_k=settings.chat.chat_search_limit)
    return hits


def _has_sufficient_evidence(hits: list[MilvusSearchHit]) -> bool:
    """Evidence-sufficiency gate: is the retrieved context strong enough to answer?

    Returns True only when the best-scoring hit clears `min_evidence_score`. When it
    doesn't (or nothing was retrieved), the caller abstains deterministically instead
    of asking the LLM to answer from weak evidence — the most reliable single control
    against grounding-gap hallucinations, since it never reaches the model at all.

    Note: `hit.distance` is the fused WeightedRanker score (dense + BM25), where higher
    means more relevant; the top hit is the max, so we only need to check it.
    """
    if not hits:
        return False
    best_score = max(hit.distance for hit in hits)
    return best_score >= settings.chat.min_evidence_score


def _apply_groundedness_check(
    response: str,
    hits: list[MilvusSearchHit],
    sources: list[ChatSource],
    llm_client: LLMClient,
) -> tuple[str, list[ChatSource], GroundednessReport | None]:
    """Run the post-generation groundedness verifier and apply the abstention policy.

    Returns the (possibly replaced) answer, its sources, and the report. When the check
    is disabled, returns the inputs unchanged with a None report. When the verifier finds
    unsupported claims and `abstain_on_ungrounded` is set, the answer is swapped for the
    canonical abstention and sources cleared; otherwise the answer stands but the report
    flags `is_grounded=False` for the caller to surface.
    """
    if not settings.chat.enable_groundedness_check:
        return response, sources, None

    report = verify_groundedness(response, hits, llm_client)
    if report.is_grounded:
        # Resolve claim → source citations and (optionally) render them into the answer.
        report.citations = resolve_citations(report.claims, sources)
        if settings.chat.render_claim_citations:
            response = render_cited_answer(response, report.citations)
        return response, sources, report

    if settings.chat.abstain_on_ungrounded:
        logger.info("Answer failed groundedness check; abstaining. checked=%s", report.checked)
        return NO_INFO_ANSWER, [], report

    logger.info("Answer failed groundedness check; returning as-is with is_grounded=False.")
    return response, sources, report


def _apply_contradiction_check(
    response: str,
    hits: list[MilvusSearchHit],
    sources: list[ChatSource],
    llm_client: LLMClient,
) -> tuple[str, ContradictionReport | None]:
    """Scan retrieved chunks for contradictions and (optionally) disclose them in the answer.

    Returns the (possibly prefixed) answer and the report. When disabled, returns the answer
    unchanged with a None report. When conflicts are found and `disclose_contradictions` is
    set, a short notice is prepended so the user knows the sources disagree; the report is
    attached to the response either way.
    """
    if not settings.chat.enable_contradiction_check:
        return response, None

    report = detect_contradictions(hits, sources, llm_client)
    if report.has_contradiction:
        logger.info("Contradiction detected among retrieved chunks: %d pair(s).", len(report.contradictions))
        if settings.chat.disclose_contradictions:
            response = render_contradiction_notice(response, report)
    return response, report


def _build_sources_and_context(
    hits: list[MilvusSearchHit],
    topic: TopicEnum,
    corpus_storage: CorpusStorageService,
) -> tuple[str, list[ChatSource]]:
    """Build the RAG context string and source reference list from Milvus hits.

    Presigned URL generation is best-effort; failures are logged and the source
    is still included with an empty URL rather than silently dropped.
    """
    context_parts: list[str] = []
    sources: list[ChatSource] = []

    for hit in hits:
        context_parts.append(f"[Source: {hit.file_name}, Page {hit.page_num}]\n{hit.chunk_text}")
        doc_url = corpus_storage.presigned_url_for(hit.doc_id, topic)
        sources.append(
            ChatSource(
                doc_id=hit.doc_id,
                file_name=hit.file_name,
                page_num=hit.page_num,
                doc_url=doc_url,
                relevance_score=hit.distance,
                chunk_excerpt=hit.chunk_text[: settings.chat.chunk_excerpt_chars],
            )
        )

    context = "\n\n---\n\n".join(context_parts) if context_parts else "No relevant documents found."
    return context, sources


def _build_chat_messages(
    system_prompt: str,
    chat_history: list[dict],
    query: str,
) -> list[ChatMessage]:
    """
    Assemble the message list for the LLM:
      [system-as-user, ...history, current-user-query]
    """
    messages = [ChatMessage(role="user", content=system_prompt)]
    for item in chat_history:
        messages.append(ChatMessage(role=item["role"], content=item["content"]))
    messages.append(ChatMessage(role="user", content=query))
    return messages


def _useful_history(chat_history: list[dict[str, object]]) -> list[dict[str, object]]:
    """Drop user→assistant pairs whose assistant turn produced no citations.

    A no-info turn (empty `metadata.sources`) carries no signal for the LLM,
    so feeding it back as context just clutters the conversation.
    """
    useful: list[dict[str, object]] = []
    pending_user: dict[str, object] | None = None
    for item in chat_history:
        if item["role"] == "user":
            pending_user = item
            continue
        sources = (item.get("metadata") or {}).get("sources") or []
        if pending_user is not None and len(sources) > 0:
            useful.append(pending_user)
            useful.append(item)
        pending_user = None
    return useful


def _build_history(raw_history: list[dict[str, object]]) -> list[ChatMessage]:
    """Build the user-facing chat history, attaching stored sources to assistant turns."""
    history: list[ChatMessage] = []
    for item in raw_history:
        sources: list[ChatSource] | None = None
        stored = (item.get("metadata") or {}).get("sources")
        if stored:
            sources = [ChatSource(**src) for src in stored]
        history.append(ChatMessage(role=item["role"], content=item["content"], sources=sources))
    return history


# LLM Chat Endpoint
@router.post(
    "/document/chat",
    response_model=ChatResponse,
    description="""
Chat with the LLM using corpus-grounded answers and optional session-based memory.

- Answers are generated **strictly from the ingested corpus documents** (RAG).
- If the corpus does not contain relevant information, the chatbot will decline to answer.
- Each response includes **source references** with document links and page numbers.

- If a **session_id is provided**, the API will retrieve recent conversation history
  from Redis and include it as context for generating the response (stateful behavior).
- If **no session_id is provided**, a new session is created automatically.
""",
)
async def chat_with_llm(
    query: str = Query(..., min_length=1, max_length=2000, description="User query"),  # noqa: B008
    topic: TopicEnum = Query(..., description="Topic to search within the corpus"),  # noqa: B008
    session_id: str = None,
    current_user: AuthenticatedKeycloakUser = Depends(require_keycloak_user),  # noqa: B008
    redis_client: RedisClient = Depends(get_redis_client),  # noqa: B008
    llm_client: LLMClient = Depends(get_llm_client),  # noqa: B008
    milvus_client: MilvusClient = Depends(get_milvus_client),  # noqa: B008
    corpus_storage: CorpusStorageService = Depends(get_corpus_storage),  # noqa: B008
    reranker: CrossEncoderReranker = Depends(get_reranker),  # noqa: B008
) -> ChatResponse:
    is_new = session_id is None
    if session_id is None:
        session_id = str(uuid.uuid4())

    logger.info(
        "document/chat called by user_id=%s username=%s topic=%s session_id=%s",
        current_user.user_id,
        current_user.username,
        topic,
        session_id,
    )

    if not is_new:
        meta = redis_client.get_session_meta(current_user.username, session_id)
        if meta is None:
            raise NotFoundError("Session not found.")

    hits = _search_corpus(query, topic, llm_client, milvus_client, reranker)

    # Evidence-sufficiency gate: if retrieval is too weak, abstain deterministically
    # without ever calling the LLM. This closes the grounding gap the prompt alone
    # can't guarantee — no evidence, no answer, no chance to hallucinate.
    groundedness: GroundednessReport | None = None
    contradiction: ContradictionReport | None = None
    if not _has_sufficient_evidence(hits):
        logger.info("Insufficient evidence for query; abstaining. session_id=%s topic=%s", session_id, topic)
        response = NO_INFO_ANSWER
        sources: list[ChatSource] = []
    else:
        context, sources = _build_sources_and_context(hits, topic, corpus_storage)
        system_prompt = CHATBOT_SYSTEM_PROMPT.format(context=context)
        chat_history = redis_client.get_recent_messages(current_user.username, session_id, settings.chat.history_context_k)
        messages = _build_chat_messages(system_prompt, _useful_history(chat_history), query)

        # Grounded path runs at grounded_temperature (0.0) so the answer stays pinned
        # to the retrieved evidence rather than sampling around it.
        response = llm_client.chat(messages, temperature=llm_client.grounded_temperature)
        if NO_INFO_DETECTOR in response.lower():
            sources = []
        else:
            # Post-generation verification: catch confident fabrications the prompt
            # can't. Only runs on real answers (a refusal has nothing to verify).
            response, sources, groundedness = _apply_groundedness_check(response, hits, sources, llm_client)
            # Surface disagreement among the retrieved sources rather than silently
            # answering from whichever chunk ranked higher. Skipped once we've abstained.
            if NO_INFO_DETECTOR not in response.lower():
                response, contradiction = _apply_contradiction_check(response, hits, sources, llm_client)

    redis_client.store_message(
        current_user.username,
        session_id,
        query,
        response,
        sources=[s.model_dump() for s in sources],
    )
    redis_client.register_session(
        username=current_user.username,
        session_id=session_id,
        topic=topic.value,
        first_query=query,
        is_new=is_new,
    )

    raw_history = redis_client.get_recent_messages(current_user.username, session_id, settings.redis.chat_history_limit)
    history = _build_history(raw_history)

    return ChatResponse(
        answer=response,
        sources=sources,
        session_id=session_id,
        history=history,
        groundedness=groundedness,
        contradiction=contradiction,
    )


@router.delete(
    "/document/session",
    response_model=str,
    description="""
Clear a conversation session from Redis.

- Deletes all stored messages associated with the provided `session_id`.
- This does NOT delete the Redis index, only the session-specific data.
""",
)
async def clear_session(
    session_id: str = Query(..., description="Session ID whose conversation history should be cleared"),  # noqa: B008
    current_user: AuthenticatedKeycloakUser = Depends(require_keycloak_user),  # noqa: B008
    redis_client: RedisClient = Depends(get_redis_client),  # noqa: B008
) -> str:
    logger.info(
        "document/session DELETE called by user_id=%s username=%s session_id=%s",
        current_user.user_id,
        current_user.username,
        session_id,
    )

    meta = redis_client.get_session_meta(current_user.username, session_id)
    if meta is None:
        raise NotFoundError("Session not found.")

    redis_client.delete_session(current_user.username, session_id)

    return f"Session '{session_id}' deleted successfully."


@router.get(
    "/document/sessions",
    response_model=UserSessionListResponse,
    description=("List the most recent chat sessions for the currently authenticated user, newest first. Capped at 20 sessions by design — older sessions are not returned."),
)
async def list_user_sessions(
    current_user: AuthenticatedKeycloakUser = Depends(require_keycloak_user),  # noqa: B008
    redis_client: RedisClient = Depends(get_redis_client),  # noqa: B008
) -> UserSessionListResponse:
    logger.info(
        "document/sessions GET called by username=%s",
        current_user.username,
    )
    raw = redis_client.list_user_sessions(current_user.username)
    sessions = [
        SessionSummary(
            session_id=s["session_id"],
            topic=s["topic"],
            created_at=s["created_at"],
            last_updated=s["last_updated"],
            message_count=int(s.get("message_count", 0)),
            preview=s.get("preview", ""),
        )
        for s in raw
    ]
    return UserSessionListResponse(sessions=sessions, total=len(sessions))


@router.get(
    "/document/sessions/{session_id}",
    response_model=SessionDetailResponse,
    description="Retrieve the full message history for a session owned by the current user.",
)
async def get_session_detail(
    session_id: str,
    current_user: AuthenticatedKeycloakUser = Depends(require_keycloak_user),  # noqa: B008
    redis_client: RedisClient = Depends(get_redis_client),  # noqa: B008
) -> SessionDetailResponse:
    logger.info(
        "document/sessions/%s GET called by username=%s",
        session_id,
        current_user.username,
    )
    meta = redis_client.get_session_meta(current_user.username, session_id)
    if meta is None:
        raise NotFoundError("Session not found.")

    raw_history = redis_client.get_recent_messages(current_user.username, session_id, settings.redis.chat_history_limit)
    history = _build_history(raw_history)
    return SessionDetailResponse(
        session_id=session_id,
        topic=meta["topic"],
        created_at=meta["created_at"],
        last_updated=meta["last_updated"],
        message_count=int(meta.get("message_count", 0)),
        preview=meta.get("preview", ""),
        history=history,
    )
