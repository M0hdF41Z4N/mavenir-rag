# Unit tests for the document score aggregation (B5). Pure numeric logic, no services.
from api.utils.scoring import aggregate_doc_score


def test_max_takes_best_chunk() -> None:
    # One excellent chunk is not diluted by weak ones — the whole point of switching off mean.
    assert aggregate_doc_score([0.1, 0.2, 0.95], aggregation="max", top_k=3) == 0.95


def test_mean_is_legacy_average() -> None:
    assert aggregate_doc_score([0.0, 1.0], aggregation="mean", top_k=3) == 0.5


def test_top_k_mean_averages_strongest_k() -> None:
    # top-2 of [0.1, 0.9, 0.5, 0.3] = 0.9, 0.5 → mean 0.7
    assert aggregate_doc_score([0.1, 0.9, 0.5, 0.3], aggregation="top_k_mean", top_k=2) == 0.7


def test_top_k_mean_k_larger_than_list_uses_all() -> None:
    assert aggregate_doc_score([0.2, 0.4], aggregation="top_k_mean", top_k=10) == 0.3


def test_empty_scores_is_zero() -> None:
    assert aggregate_doc_score([], aggregation="max", top_k=3) == 0.0


def test_max_beats_mean_for_one_strong_chunk() -> None:
    # Regression intent: the doc with one strong chunk should now outscore a uniformly
    # mediocre doc, which mean averaging failed to distinguish.
    strong_doc = aggregate_doc_score([0.95, 0.1, 0.1], aggregation="max", top_k=3)
    mediocre_doc = aggregate_doc_score([0.5, 0.5, 0.5], aggregation="max", top_k=3)
    assert strong_doc > mediocre_doc
