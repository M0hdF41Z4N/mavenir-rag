# Unit tests for the contradiction detector (B6). LLM boundary mocked; no live services.
from unittest.mock import MagicMock

from api.models import ChatSource, ContradictionReport, MilvusSearchHit
from api.services.contradiction_detector import (
    detect_contradictions,
    render_contradiction_notice,
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


def test_fewer_than_two_chunks_cannot_contradict() -> None:
    llm = _llm_returning("{}")
    report = detect_contradictions([_hit(1, "only one")], [_source(1)], llm)
    assert report.checked is True
    assert report.has_contradiction is False
    llm.chat.assert_not_called()  # short-circuits before any LLM call


def test_no_contradictions_reported() -> None:
    payload = '{"contradictions": []}'
    report = detect_contradictions([_hit(1, "a"), _hit(2, "b")], [_source(1), _source(2)], _llm_returning(payload))
    assert report.checked is True
    assert report.has_contradiction is False


def test_contradiction_detected_and_sources_resolved() -> None:
    payload = (
        '{"contradictions": [{"source_index_a": 1, "source_index_b": 2, '
        '"description": "CEO named differently"}]}'
    )
    report = detect_contradictions([_hit(1, "CEO is Alice"), _hit(2, "CEO is Bob")], [_source(1), _source(2)], _llm_returning(payload))
    assert report.has_contradiction is True
    c = report.contradictions[0]
    assert c.source_a.doc_id == "doc-1"
    assert c.source_b.doc_id == "doc-2"


def test_out_of_range_pair_is_dropped() -> None:
    # Only 2 chunks, but the detector cites chunk 3 — a fabricated reference, discarded.
    payload = '{"contradictions": [{"source_index_a": 1, "source_index_b": 3, "description": "x"}]}'
    report = detect_contradictions([_hit(1, "a"), _hit(2, "b")], [_source(1), _source(2)], _llm_returning(payload))
    assert report.has_contradiction is False


def test_self_referential_pair_is_dropped() -> None:
    payload = '{"contradictions": [{"source_index_a": 1, "source_index_b": 1, "description": "x"}]}'
    report = detect_contradictions([_hit(1, "a"), _hit(2, "b")], [_source(1), _source(2)], _llm_returning(payload))
    assert report.has_contradiction is False


def test_detector_uses_verifier_model_at_temp_zero() -> None:
    payload = '{"contradictions": []}'
    llm = _llm_returning(payload)
    detect_contradictions([_hit(1, "a"), _hit(2, "b")], [_source(1), _source(2)], llm)
    _, kwargs = llm.chat.call_args
    assert kwargs["model"] == "ollama/llama3.1:8b"
    assert kwargs["temperature"] == 0.0


def test_detector_fails_safe_on_bad_json() -> None:
    report = detect_contradictions([_hit(1, "a"), _hit(2, "b")], [_source(1), _source(2)], _llm_returning("garbage"))
    assert report.checked is False
    assert report.has_contradiction is False


def test_detector_fails_safe_on_llm_exception() -> None:
    llm = MagicMock()
    llm.grounded_temperature = 0.0
    llm.verifier_model = "ollama/llama3.1:8b"
    llm.chat.side_effect = RuntimeError("provider down")
    report = detect_contradictions([_hit(1, "a"), _hit(2, "b")], [_source(1), _source(2)], llm)
    assert report.checked is False
    assert report.has_contradiction is False


def test_render_notice_prepends_when_contradiction() -> None:
    report = ContradictionReport(
        checked=True,
        has_contradiction=True,
        contradictions=[],
    )
    # Build a contradiction with a description for the notice.
    from api.models import Contradiction

    report.contradictions = [Contradiction(source_index_a=1, source_index_b=2, description="sources disagree on X")]
    rendered = render_contradiction_notice("The answer.", report)
    assert rendered.startswith("Note: the retrieved sources disagree")
    assert "sources disagree on X" in rendered
    assert rendered.endswith("The answer.")


def test_render_notice_noop_without_contradiction() -> None:
    report = ContradictionReport(checked=True, has_contradiction=False, contradictions=[])
    assert render_contradiction_notice("unchanged", report) == "unchanged"
