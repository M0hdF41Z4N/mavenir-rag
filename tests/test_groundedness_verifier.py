# Unit tests for the post-generation groundedness verifier. The LLM boundary is mocked,
# so these run without live services.
from unittest.mock import MagicMock

from api.models import ChatSource, ClaimVerdict, MilvusSearchHit
from api.services.groundedness_verifier import (
    build_numbered_context,
    render_cited_answer,
    resolve_citations,
    verify_groundedness,
)


def _hit(index: int, text: str) -> MilvusSearchHit:
    return MilvusSearchHit(
        id=index,
        distance=0.9,
        doc_id=f"doc-{index}",
        file_name=f"doc-{index}.pdf",
        chunk_text=text,
        parent_headings=["Section"],
        page_num=index,
    )


def _source(index: int) -> ChatSource:
    return ChatSource(
        doc_id=f"doc-{index}",
        file_name=f"doc-{index}.pdf",
        page_num=index,
        doc_url=f"https://example/{index}",
        relevance_score=0.9,
        chunk_excerpt=f"excerpt {index}",
    )


def _llm_returning(payload: str) -> MagicMock:
    llm = MagicMock()
    llm.grounded_temperature = 0.0
    llm.verifier_model = "ollama/llama3.1:8b"
    llm.chat.return_value = payload
    return llm


def test_numbered_context_uses_one_based_markers() -> None:
    hits = [_hit(1, "Alpha fact."), _hit(2, "Beta fact.")]
    context = build_numbered_context(hits)
    assert "[1] (Source: doc-1.pdf, Page 1)" in context
    assert "[2] (Source: doc-2.pdf, Page 2)" in context
    assert "Alpha fact." in context and "Beta fact." in context


def test_all_claims_supported_is_grounded() -> None:
    payload = (
        '{"claims": ['
        '{"claim": "Alpha is true", "source_indices": [1], "is_supported": true},'
        '{"claim": "Beta is true", "source_indices": [2], "is_supported": true}'
        "]}"
    )
    report = verify_groundedness("answer", [_hit(1, "a"), _hit(2, "b")], _llm_returning(payload))
    assert report.checked is True
    assert report.is_grounded is True
    assert len(report.claims) == 2
    assert report.claims[0].source_indices == [1]


def test_any_unsupported_claim_is_not_grounded() -> None:
    payload = (
        '{"claims": ['
        '{"claim": "Alpha is true", "source_indices": [1], "is_supported": true},'
        '{"claim": "Gamma is invented", "source_indices": [], "is_supported": false}'
        "]}"
    )
    report = verify_groundedness("answer", [_hit(1, "a")], _llm_returning(payload))
    assert report.checked is True
    assert report.is_grounded is False


def test_fence_wrapped_json_is_parsed() -> None:
    payload = '```json\n{"claims": [{"claim": "x", "source_indices": [1], "is_supported": true}]}\n```'
    report = verify_groundedness("answer", [_hit(1, "a")], _llm_returning(payload))
    assert report.checked is True
    assert report.is_grounded is True


def test_verifier_fails_closed_on_bad_json() -> None:
    # Malformed output must NOT be treated as a passing verification.
    report = verify_groundedness("answer", [_hit(1, "a")], _llm_returning("not json at all"))
    assert report.checked is False
    assert report.is_grounded is False
    assert report.claims == []


def test_verifier_fails_closed_on_llm_exception() -> None:
    llm = MagicMock()
    llm.grounded_temperature = 0.0
    llm.verifier_model = "ollama/llama3.1:8b"
    llm.chat.side_effect = RuntimeError("provider down")
    report = verify_groundedness("answer", [_hit(1, "a")], llm)
    assert report.checked is False
    assert report.is_grounded is False


def test_verifier_uses_separate_verifier_model() -> None:
    # The check must call the configured verifier_model, not the generation model, so it
    # is not self-judging.
    payload = '{"claims": [{"claim": "x", "source_indices": [1], "is_supported": true}]}'
    llm = _llm_returning(payload)
    verify_groundedness("answer", [_hit(1, "a")], llm)
    _, kwargs = llm.chat.call_args
    assert kwargs["model"] == "ollama/llama3.1:8b"
    assert kwargs["temperature"] == 0.0


# --- B4: citation index validation & rendering ---


def test_out_of_range_citation_is_dropped_and_downgraded() -> None:
    # Only one chunk exists, but the verifier cites [2] — a fabricated source. The index
    # is dropped and, with nothing left to back it, the claim is downgraded to unsupported.
    payload = '{"claims": [{"claim": "invented cite", "source_indices": [2], "is_supported": true}]}'
    report = verify_groundedness("answer", [_hit(1, "a")], _llm_returning(payload))
    assert report.claims[0].source_indices == []
    assert report.claims[0].is_supported is False
    assert report.is_grounded is False


def test_valid_and_invalid_indices_mixed_keeps_only_valid() -> None:
    payload = '{"claims": [{"claim": "c", "source_indices": [1, 5, 1], "is_supported": true}]}'
    report = verify_groundedness("answer", [_hit(1, "a"), _hit(2, "b")], _llm_returning(payload))
    # 5 dropped (out of range), duplicate 1 collapsed; claim stays supported via [1].
    assert report.claims[0].source_indices == [1]
    assert report.claims[0].is_supported is True
    assert report.is_grounded is True


def test_resolve_citations_maps_indices_to_sources() -> None:
    claims = [
        ClaimVerdict(claim="c1", source_indices=[1], is_supported=True),
        ClaimVerdict(claim="c2", source_indices=[], is_supported=False),
    ]
    sources = [_source(1), _source(2)]
    citations = resolve_citations(claims, sources)
    assert citations[0].sources[0].doc_id == "doc-1"
    assert citations[1].sources == []


def test_render_cited_answer_appends_per_claim_markers() -> None:
    claims = [ClaimVerdict(claim="Sky is blue", source_indices=[1, 2], is_supported=True)]
    citations = resolve_citations(claims, [_source(1), _source(2)])
    rendered = render_cited_answer("Sky is blue.", citations)
    assert "Claims & sources:" in rendered
    assert "- Sky is blue [1][2]" in rendered


def test_render_cited_answer_noop_without_citations() -> None:
    assert render_cited_answer("just this", []) == "just this"
