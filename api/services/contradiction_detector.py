# Contradiction detector.
#
# Scans the retrieved chunks for DIRECT contradictions (two chunks making mutually
# exclusive factual claims about the same thing) via a separate LLM pass, and reports the
# conflicting pairs. Silently answering from whichever chunk happened to rank higher hides
# real disagreement in the corpus; surfacing it is safer.
#
# Like the groundedness verifier, this runs on the configured verifier_model, validates the
# returned indices against the real chunk set, and fails SAFE: any error reports
# checked=False / has_contradiction=False so a failed check never fabricates a conflict.
import json
import logging
import re

from api.client.llm_client import LLMClient
from api.models import (
    ChatMessage,
    ChatSource,
    Contradiction,
    ContradictionReport,
    MilvusSearchHit,
)
from api.services.groundedness_verifier import build_numbered_context
from api.utils.llm_prompts import CONTRADICTION_DETECTOR_PROMPT

logger = logging.getLogger(__name__)


def _parse_contradictions(response: str) -> list[Contradiction]:
    """Parse the detector's strict-JSON output. Raises on any structural problem."""
    cleaned = re.sub(r"^```(?:json)?\s*", "", response.strip())
    cleaned = re.sub(r"\s*```$", "", cleaned)
    data = json.loads(cleaned)
    return [Contradiction(**item) for item in data["contradictions"]]


def _keep_valid(contradictions: list[Contradiction], num_chunks: int) -> list[Contradiction]:
    """Drop pairs that reference a non-existent chunk or the same chunk twice.

    A fabricated index is not a real contradiction, so we discard it deterministically
    rather than trust the detector LLM to only cite real chunks.
    """
    valid_range = range(1, num_chunks + 1)
    kept: list[Contradiction] = []
    for item in contradictions:
        if item.source_index_a == item.source_index_b:
            continue
        if item.source_index_a in valid_range and item.source_index_b in valid_range:
            kept.append(item)
    return kept


def _resolve_sources(contradictions: list[Contradiction], sources: list[ChatSource]) -> list[Contradiction]:
    """Attach the resolved ChatSource for each side of a contradiction, when available."""
    resolved: list[Contradiction] = []
    for item in contradictions:
        source_a = sources[item.source_index_a - 1] if item.source_index_a <= len(sources) else None
        source_b = sources[item.source_index_b - 1] if item.source_index_b <= len(sources) else None
        resolved.append(
            Contradiction(
                source_index_a=item.source_index_a,
                source_index_b=item.source_index_b,
                description=item.description,
                source_a=source_a,
                source_b=source_b,
            )
        )
    return resolved


def detect_contradictions(
    hits: list[MilvusSearchHit],
    sources: list[ChatSource],
    llm_client: LLMClient,
) -> ContradictionReport:
    """Report DIRECT contradictions among the retrieved chunks.

    Fewer than two chunks can't contradict, so that short-circuits to a clean report. On
    any detector failure the report is fail-safe (checked=False, has_contradiction=False).
    """
    if len(hits) < 2:
        return ContradictionReport(checked=True, has_contradiction=False, contradictions=[])

    numbered_context = build_numbered_context(hits)
    prompt = CONTRADICTION_DETECTOR_PROMPT.format(numbered_context=numbered_context)

    try:
        response = llm_client.chat(
            [ChatMessage(role="user", content=prompt)],
            temperature=llm_client.grounded_temperature,
            model=llm_client.verifier_model,
        )
        contradictions = _parse_contradictions(response)
    except Exception:
        logger.exception("Contradiction detector failed; reporting no contradiction (fail-safe).")
        return ContradictionReport(checked=False, has_contradiction=False, contradictions=[])

    contradictions = _keep_valid(contradictions, num_chunks=len(hits))
    contradictions = _resolve_sources(contradictions, sources)
    return ContradictionReport(
        checked=True,
        has_contradiction=len(contradictions) > 0,
        contradictions=contradictions,
    )


def render_contradiction_notice(answer: str, report: ContradictionReport) -> str:
    """Prepend a short disclosure when sources disagree; return the answer unchanged otherwise."""
    if not report.has_contradiction:
        return answer
    notice = "Note: the retrieved sources disagree on some points:"
    lines = [notice] + [f"- {c.description}" for c in report.contradictions]
    return "\n".join(lines) + "\n\n" + answer.strip()
