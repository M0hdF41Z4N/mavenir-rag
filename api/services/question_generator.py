# Generates Q&A pairs for matched corpus chunks using the LLM.
# Called as part of the inference pipeline after Milvus search returns chunk hits.
# The four question types (subjective, MCQ, match-making, one-word) each have a
# dedicated prompt template in llm_prompts.py; this module dispatches to the right one.
import json
import logging
import re
from dataclasses import dataclass

from api.client.llm_client import LLMClient
from api.client.minio_client import MinioClient
from api.client.redis_client import RedisClient
from api.models import (
    ChatMessage,
    InputQueryOverallReport,
    KnowledgeFeedModeEnum,
    LLMGeneratedResponse,
    QuestionTypeEnum,
)
from api.utils.llm_prompts import (
    MATCH_PROMPT_TEMPLATE,
    MCQ_PROMPT_TEMPLATE,
    ONE_WORD_PROMPT_TEMPLATE,
    SUBJECTIVE_PROMPT_TEMPLATE,
)

logger = logging.getLogger(__name__)


@dataclass
class QuestionGenerationParams:
    username: str
    session_id: str
    question_answer_type: QuestionTypeEnum
    knowledge_feed_mode: KnowledgeFeedModeEnum
    total_questions: int
    llm_client: LLMClient
    minio_client: MinioClient
    redis_client: RedisClient


# Dispatch table: maps each question type to its prompt template.
# Adding a new question type only requires adding the template in llm_prompts.py and
# a new entry here — the rest of the pipeline is generic.
_PROMPT_BY_TYPE: dict[QuestionTypeEnum, str] = {
    QuestionTypeEnum.SUBJECTIVE: SUBJECTIVE_PROMPT_TEMPLATE,
    QuestionTypeEnum.MCQ: MCQ_PROMPT_TEMPLATE,
    QuestionTypeEnum.MATCH_MAKING: MATCH_PROMPT_TEMPLATE,
    QuestionTypeEnum.ONE_WORD: ONE_WORD_PROMPT_TEMPLATE,
}


def _parse_llm_response(response: str) -> LLMGeneratedResponse:
    """Parse LLM response into LLMGeneratedResponse, falling back to raw string on failure."""
    try:
        cleaned = re.sub(r"^```(?:json)?\s*", "", response.strip())
        cleaned = re.sub(r"\s*```$", "", cleaned)
        data = json.loads(cleaned)
        return LLMGeneratedResponse(**data)
    except (json.JSONDecodeError, Exception) as e:
        logger.warning("Failed to parse LLM response as JSON: %s", e)
        return LLMGeneratedResponse(raw=response)


def generate_questions_from_chunks(
    milvus_reports: list[InputQueryOverallReport],
    params: QuestionGenerationParams,
) -> list[InputQueryOverallReport]:
    """Generate question-answer pairs for each corpus chunk using the LLM."""
    prompt = _PROMPT_BY_TYPE.get(params.question_answer_type, SUBJECTIVE_PROMPT_TEMPLATE)

    for input_query_report in milvus_reports:
        for doc in input_query_report.detailed_matches:
            for chunk in doc.chunk_matches:
                # Only TEXT and HYBRID modes feed chunk text to the LLM.
                # IMAGE mode is intended to send page images instead, but multimodal
                # question generation is not yet implemented — skip the chunk entirely.
                if params.knowledge_feed_mode not in (
                    KnowledgeFeedModeEnum.TEXT,
                    KnowledgeFeedModeEnum.HYBRID,
                ):
                    continue

                # Store a short summary (not the full prompt) in Redis — the full prompt is too large for chat history
                llm_query_summary = f"Generate Question-Answer pairs for the following chunk: {chunk.corpus_chunk}"
                prompt_feed = prompt.format(total_questions=params.total_questions, corpus_text=chunk.corpus_chunk)

                try:
                    response = params.llm_client.chat([ChatMessage(role="user", content=prompt_feed)])
                    chunk.llm_generated_questions = _parse_llm_response(response)
                    params.redis_client.store_message(params.username, params.session_id, llm_query_summary, response)
                except Exception as e:
                    chunk.llm_generated_questions = LLMGeneratedResponse(raw=f"Failed to generate questions: {e}")

    return milvus_reports
