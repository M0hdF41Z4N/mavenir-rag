# RAG — Missing Components TODO

Gaps identified by comparing the target RAG architecture against the current codebase.

## 1. Streamlit chat interface
- [ ] Build a Streamlit frontend wired to the existing `/document/chat` endpoint ([matcher_router.py:248](api/routers/matcher_router.py#L248)).
- **Status:** Completely absent — no `streamlit` anywhere in the repo. Backend chat exists; no UI.

## 2. DOCX / TXT ingestion
- [ ] Add real per-format parsing for `.docx` and `.txt` in [hybrid_parser.py](api/utils/parsers/hybrid_parser.py).
- **Status:** Config allows `.pdf/.docx/.txt/.md` ([config.py:70](api/config.py#L70)), but the parser hardcodes PDF (`pymupdf.open(..., filetype="pdf")` at [hybrid_parser.py:133](api/utils/parsers/hybrid_parser.py#L133); only `InputFormat.PDF` registered with docling). DOCX/TXT fail or misparse.

## 3. RRF hybrid reranking
- [ ] Add `RRFRanker` as a selectable reranking strategy alongside the existing `WeightedRanker`.
- **Status:** Only `WeightedRanker` is wired up ([milvus_client.py:319](api/client/milvus_client.py#L319)). RRF is neither imported nor config-selectable.

## 4. HuggingFace embedding provider
- [ ] Add a HuggingFace / sentence-transformers dense-embedding path.
- **Status:** Provider enum supports `ollama`/`openai`/`gemini` ([config.py:23](api/config.py#L23)); HuggingFace (listed in the target architecture) is missing.
