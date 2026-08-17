# Post-generation groundedness verifier.
#
# After the grounded chat path produces an answer, this runs a SEPARATE LLM pass that
# decomposes the answer into atomic claims and checks each against the retrieved chunks.
# A distinct verifier pass is a stronger control than trusting the generator to police
# itself: it catches confident fabrications that the generation prompt alone cannot.
#
# The verifier is best-effort — if the LLM call or JSON parse fails, it reports
# checked=False and is_grounded=False (fail-closed), so callers never treat an
# unverified answer as verified.
import json
import logging
import re

from api.client.llm_client import LLMClient
from api.models import (
    ChatMessage,
    ChatSource,
    ClaimCitation,
    ClaimVerdict,
    GroundednessReport,
    MilvusSearchHit,
)
from api.utils.llm_prompts import GROUNDEDNESS_VERIFIER_PROMPT

logger = logging.getLogger(__name__)


def build_numbered_context(hits: list[MilvusSearchHit]) -> str:
    """Render retrieved chunks as ``[n]`` blocks so claims can cite them by index.

    The 1-based ``[n]`` markers are the same indices the verifier returns in
    ``source_indices``, giving a claim → chunk mapping (the basis for claim-level
    citations).
    """
    blocks: list[str] = []
    for index, hit in enumerate(hits, start=1):
        label = f"[{index}] (Source: {hit.file_name}, Page {hit.page_num})"
        blocks.append(f"{label}\n{hit.chunk_text}")
    return "\n\n---\n\n".join(blocks)


def _parse_claims(response: str) -> list[ClaimVerdict]:
    """Parse the verifier's strict-JSON output into claim verdicts.

    Mirrors the fence-stripping tolerance used elsewhere for LLM JSON. Raises on any
    structural problem so the caller can fail closed.
    """
    cleaned = re.sub(r"^```(?:json)?\s*", "", response.strip())
    cleaned = re.sub(r"\s*```$", "", cleaned)
    data = json.loads(cleaned)
    return [ClaimVerdict(**claim) for claim in data["claims"]]


def _validate_citations(claims: list[ClaimVerdict], num_chunks: int) -> list[ClaimVerdict]:
    """Deterministically enforce that every cited index references a real chunk.

    A citation pointing outside ``1..num_chunks`` is a fabricated source — the model
    invented a chunk number. We drop such indices programmatically (not trusting the
    verifier LLM to police itself), deduplicate while preserving order, and downgrade any
    claim that ends up ``is_supported`` with no valid citation left to unsupported.
    """
    valid_range = range(1, num_chunks + 1)
    validated: list[ClaimVerdict] = []
    for claim in claims:
        kept: list[int] = []
        for index in claim.source_indices:
            if index in valid_range and index not in kept:
                kept.append(index)
        # A claim can only stay "supported" if a real chunk still backs it.
        is_supported = claim.is_supported and len(kept) > 0
        validated.append(ClaimVerdict(claim=claim.claim, source_indices=kept, is_supported=is_supported))
    return validated


def verify_groundedness(
    answer: str,
    hits: list[MilvusSearchHit],
    llm_client: LLMClient,
) -> GroundednessReport:
    """Check that every factual claim in ``answer`` is entailed by the retrieved chunks.

    Returns a report with per-claim verdicts. ``is_grounded`` is True only when every
    extracted claim is supported. On any verifier failure the report is fail-closed
    (checked=False, is_grounded=False) so an unverified answer is never mistaken for a
    verified one.
    """
    numbered_context = build_numbered_context(hits)
    prompt = GROUNDEDNESS_VERIFIER_PROMPT.format(numbered_context=numbered_context, answer=answer)

    try:
        # Run the verifier deterministically (grounded_temperature = 0.0), as an
        # independent pass (fresh messages, no generation history), and on the configured
        # verifier_model so a separate model — not the generator — grades the answer.
        response = llm_client.chat(
            [ChatMessage(role="user", content=prompt)],
            temperature=llm_client.grounded_temperature,
            model=llm_client.verifier_model,
        )
        claims = _parse_claims(response)
    except Exception:
        logger.exception("Groundedness verifier failed; treating answer as unverified.")
        return GroundednessReport(checked=False, is_grounded=False, claims=[])

    # Programmatically reject fabricated chunk numbers before trusting the verdict.
    claims = _validate_citations(claims, num_chunks=len(hits))
    is_grounded = all(claim.is_supported for claim in claims)
    return GroundednessReport(checked=True, is_grounded=is_grounded, claims=claims)


def resolve_citations(claims: list[ClaimVerdict], sources: list[ChatSource]) -> list[ClaimCitation]:
    """Map each claim's validated 1-based indices to the corresponding ``ChatSource``.

    ``sources`` must be index-aligned with the numbered context (``sources[n-1]`` is chunk
    ``[n]``) — which is how the chat endpoint builds them. Indices are already validated
    upstream, so every index here resolves to a real source.
    """
    citations: list[ClaimCitation] = []
    for claim in claims:
        resolved = [sources[index - 1] for index in claim.source_indices if 1 <= index <= len(sources)]
        citations.append(
            ClaimCitation(
                claim=claim.claim,
                source_indices=claim.source_indices,
                sources=resolved,
                is_supported=claim.is_supported,
            )
        )
    return citations


def render_cited_answer(answer: str, citations: list[ClaimCitation]) -> str:
    """Append a per-claim citation breakdown to the answer for claim-level attribution.

    Rather than surgically splicing ``[n]`` into the model's prose (brittle), we keep the
    original answer and add a structured "Claims & sources" section mapping each claim to
    its ``[n]`` markers, so attribution is explicit and verifiable. Returns the answer
    unchanged when there are no citations to show.
    """
    if not citations:
        return answer

    lines = [answer.strip(), "", "Claims & sources:"]
    for citation in citations:
        markers = "".join(f"[{index}]" for index in citation.source_indices) or "[unsupported]"
        lines.append(f"- {citation.claim} {markers}")
    return "\n".join(lines)
