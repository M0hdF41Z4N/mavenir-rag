# Unit tests for the cross-encoder reranker. The CrossEncoder model is mocked (no torch /
# sentence-transformers needed), so these run without the optional rerank dependency.
from unittest.mock import MagicMock

from api.config import RerankSettings
from api.models import MilvusSearchHit
from api.services.reranker import CrossEncoderReranker


def _hit(index: int, text: str) -> MilvusSearchHit:
    return MilvusSearchHit(
        id=index,
        distance=0.5,
        doc_id=f"doc-{index}",
        file_name=f"doc-{index}.pdf",
        chunk_text=text,
        parent_headings=["Section"],
        page_num=index,
    )


def _reranker_with_scores(scores: list[float], top_k: int = 5) -> CrossEncoderReranker:
    """Build a reranker whose model returns `scores` from predict(), bypassing lazy load."""
    reranker = CrossEncoderReranker(RerankSettings(rerank_top_k=top_k))
    model = MagicMock()
    model.predict.return_value = scores
    reranker._model = model  # skip the guarded import / real model load
    return reranker


def test_rerank_orders_by_cross_encoder_score() -> None:
    hits = [_hit(1, "a"), _hit(2, "b"), _hit(3, "c")]
    # Milvus order is 1,2,3 but the cross-encoder prefers 2 > 3 > 1.
    reranker = _reranker_with_scores([0.1, 0.9, 0.5])
    ranked = reranker.rerank("query", hits)
    assert [h.id for h in ranked] == [2, 3, 1]


def test_rerank_truncates_to_top_k() -> None:
    hits = [_hit(i, str(i)) for i in range(1, 6)]
    reranker = _reranker_with_scores([0.1, 0.2, 0.3, 0.4, 0.5], top_k=2)
    ranked = reranker.rerank("query", hits)
    assert [h.id for h in ranked] == [5, 4]


def test_rerank_top_k_override_wins() -> None:
    hits = [_hit(i, str(i)) for i in range(1, 4)]
    reranker = _reranker_with_scores([0.3, 0.2, 0.1], top_k=5)
    ranked = reranker.rerank("query", hits, top_k=1)
    assert [h.id for h in ranked] == [1]


def test_rerank_empty_hits_returns_empty() -> None:
    reranker = _reranker_with_scores([])
    assert reranker.rerank("query", []) == []


def test_rerank_fails_safe_when_model_unavailable() -> None:
    # No model and load already marked failed → original order, truncated to top_k.
    reranker = CrossEncoderReranker(RerankSettings(rerank_top_k=2))
    reranker._load_failed = True
    hits = [_hit(1, "a"), _hit(2, "b"), _hit(3, "c")]
    ranked = reranker.rerank("query", hits)
    assert [h.id for h in ranked] == [1, 2]


def test_rerank_fails_safe_on_predict_error() -> None:
    reranker = CrossEncoderReranker(RerankSettings(rerank_top_k=3))
    model = MagicMock()
    model.predict.side_effect = RuntimeError("scoring blew up")
    reranker._model = model
    hits = [_hit(1, "a"), _hit(2, "b")]
    ranked = reranker.rerank("query", hits)
    # Falls back to original order rather than dropping the turn.
    assert [h.id for h in ranked] == [1, 2]


def test_missing_dependency_disables_reranking(monkeypatch) -> None:
    # Simulate sentence-transformers not being installed: _get_model returns None and
    # rerank fails safe. We force the ImportError path via a fresh reranker.
    import builtins

    reranker = CrossEncoderReranker(RerankSettings(rerank_top_k=2))
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "sentence_transformers":
            raise ImportError("not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    hits = [_hit(1, "a"), _hit(2, "b"), _hit(3, "c")]
    ranked = reranker.rerank("query", hits)
    assert [h.id for h in ranked] == [1, 2]
    assert reranker._load_failed is True
