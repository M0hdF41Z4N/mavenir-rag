# Cross-encoder reranker.
#
# Sits between hybrid retrieval and the LLM: it re-scores each (query, chunk) pair with a
# cross-encoder that reads the query and chunk *together* — unlike bi-encoder/BM25 fusion,
# which scores them independently — then keeps the top-k. This is the highest-leverage
# retrieval-quality improvement available and directly reduces grounding-gap hallucinations
# by putting genuinely relevant chunks in front of the model.
#
# The CrossEncoder (sentence-transformers / torch) is a heavy optional dependency, so the
# import is guarded and the model is lazy-loaded on first use. Any failure (missing dep,
# model load error, scoring error) fails SAFE: the original hit order is returned unchanged,
# so reranking can never make retrieval worse than the fusion baseline.
import logging

from api.config import RerankSettings
from api.models import MilvusSearchHit

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Reranks Milvus hits with a sentence-transformers CrossEncoder.

    The model is loaded lazily and cached on the instance. Construct once (e.g. a
    dependency-injected singleton) and reuse across requests to avoid reloading it.
    """

    def __init__(self, settings: RerankSettings) -> None:
        self.config = settings
        self._model: object | None = None
        self._load_failed = False

    def _get_model(self) -> object | None:
        """Lazy-load and cache the CrossEncoder; return None if it can't be loaded.

        The guarded import keeps sentence-transformers/torch optional — the app runs
        without them as long as reranking stays disabled.
        """
        if self._model is not None:
            return self._model
        if self._load_failed:
            return None
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            logger.warning("sentence-transformers not installed; reranking disabled. Install it to enable.")
            self._load_failed = True
            return None
        try:
            self._model = CrossEncoder(self.config.rerank_model)
        except Exception:
            logger.exception("Failed to load rerank model %s; skipping reranking.", self.config.rerank_model)
            self._load_failed = True
            return None
        return self._model

    def rerank(self, query: str, hits: list[MilvusSearchHit], top_k: int | None = None) -> list[MilvusSearchHit]:
        """Return the ``top_k`` hits most relevant to ``query``, cross-encoder ordered.

        Fails safe: on any error, or when the model is unavailable, returns the original
        hits truncated to ``top_k`` — never worse than the fusion baseline.
        """
        keep = self.config.rerank_top_k if top_k is None else top_k
        if not hits:
            return []

        model = self._get_model()
        if model is None:
            return hits[:keep]

        try:
            pairs = [(query, hit.chunk_text or "") for hit in hits]
            scores = model.predict(pairs)
            ranked = sorted(zip(hits, scores, strict=True), key=lambda pair: pair[1], reverse=True)
            return [hit for hit, _ in ranked[:keep]]
        except Exception:
            logger.exception("Reranking failed; falling back to original hit order.")
            return hits[:keep]
