# Unit tests for the evidence-sufficiency gate and grounded-temperature wiring
# added to the /document/chat path. These exercise the pure decision logic at the
# boundary — no Redis/Milvus/LLM/Keycloak calls — so they run without live services.
from unittest.mock import MagicMock

from api.config import LLMSettings
from api.client.llm_client import LLMClient
from api.models import MilvusSearchHit
from api.routers.matcher_router import _has_sufficient_evidence


def _hit(distance: float) -> MilvusSearchHit:
    return MilvusSearchHit(
        id=1,
        distance=distance,
        doc_id="doc-1",
        file_name="doc-1.pdf",
        chunk_text="some chunk text",
        parent_headings=["Section"],
        page_num=3,
    )


def test_empty_hits_are_insufficient() -> None:
    # No retrieval → abstain, never reach the LLM.
    assert _has_sufficient_evidence([]) is False


def test_best_hit_at_or_above_floor_is_sufficient(monkeypatch) -> None:
    # Happy path: the top hit clears the configured floor.
    from api.config import settings

    monkeypatch.setattr(settings.chat, "min_evidence_score", 0.5)
    hits = [_hit(0.3), _hit(0.72), _hit(0.1)]  # max = 0.72 >= 0.5
    assert _has_sufficient_evidence(hits) is True


def test_best_hit_below_floor_is_insufficient(monkeypatch) -> None:
    # Error/abstain path: even the strongest hit is too weak.
    from api.config import settings

    monkeypatch.setattr(settings.chat, "min_evidence_score", 0.8)
    hits = [_hit(0.3), _hit(0.72), _hit(0.1)]  # max = 0.72 < 0.8
    assert _has_sufficient_evidence(hits) is False


def test_score_exactly_at_floor_is_sufficient(monkeypatch) -> None:
    # Boundary: the floor is inclusive (>=).
    from api.config import settings

    monkeypatch.setattr(settings.chat, "min_evidence_score", 0.5)
    assert _has_sufficient_evidence([_hit(0.5)]) is True


def test_grounded_chat_uses_grounded_temperature() -> None:
    # The grounded path must pass grounded_temperature (0.0) through to LiteLLM,
    # not the creative chat_temperature.
    llm = LLMClient(LLMSettings(chat_temperature=0.2, grounded_temperature=0.0))

    captured: dict[str, float] = {}

    def fake_completion(**kwargs):
        captured["temperature"] = kwargs["temperature"]
        return MagicMock(choices=[MagicMock(message=MagicMock(content="ok"))])

    import api.client.llm_client as llm_module

    original = llm_module.completion
    llm_module.completion = fake_completion
    try:
        llm.chat([], temperature=llm.grounded_temperature)
    finally:
        llm_module.completion = original

    assert captured["temperature"] == 0.0


def test_chat_model_override_is_passed_through() -> None:
    llm = LLMClient(LLMSettings(litellm_chat_model="ollama/gemma3:4b"))

    captured: dict[str, str] = {}

    def fake_completion(**kwargs):
        captured["model"] = kwargs["model"]
        return MagicMock(choices=[MagicMock(message=MagicMock(content="ok"))])

    import api.client.llm_client as llm_module

    original = llm_module.completion
    llm_module.completion = fake_completion
    try:
        llm.chat([], model="ollama/llama3.1:8b")
    finally:
        llm_module.completion = original

    assert captured["model"] == "ollama/llama3.1:8b"


def test_verifier_model_defaults_to_chat_model_when_unset() -> None:
    # Self-judging fallback: no separate verifier configured → verifier_model == chat model.
    llm = LLMClient(LLMSettings(litellm_chat_model="ollama/gemma3:4b", litellm_verifier_model=None))
    assert llm.verifier_model == "ollama/gemma3:4b"


def test_verifier_model_uses_configured_value() -> None:
    llm = LLMClient(
        LLMSettings(litellm_chat_model="ollama/gemma3:4b", litellm_verifier_model="ollama/llama3.1:8b")
    )
    assert llm.verifier_model == "ollama/llama3.1:8b"
