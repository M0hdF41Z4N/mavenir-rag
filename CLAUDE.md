# Project Guidelines

## Project context

Python FastAPI application — RAG pipeline with Redis (session storage), Milvus (vector DB), LiteLLM (LLM abstraction), Keycloak (OAuth2/OIDC auth), MinIO (object storage), and Docling/PyMuPDF for document ingestion.

---

## Coding conventions

### General

1. All public functions must have explicit type annotations on parameters and return types.
2. Use `logging` instead of `print()` in all non-CLI production code paths. CLI scripts (e.g. entry points under `api/scripts/`) may use `print()` for intentional user-facing stdout output (results, next-step instructions); use `logger` for diagnostics within those scripts.
3. Use context managers (`with`) for all file and resource management.
4. Magic numbers must be replaced with named constants.
5. Functions longer than 50 lines or with more than 5 parameters must be split or use a dataclass.
6. Nesting deeper than 4 levels must be flattened using early returns.

### Async

- Never use blocking I/O (`requests`, `time.sleep`, synchronous file reads) inside `async` route handlers.
- Use `httpx.AsyncClient` for outbound HTTP calls in async context.
- Use `asyncio.run_in_executor` for CPU-bound work inside async handlers.

### Error handling

- Never use bare `except: pass` — always catch specific exceptions and log them.
- Never swallow exceptions silently — log at minimum `logger.exception(...)`.
- Always use context managers for resources (files, DB connections, Redis clients).

### Imports

- Use only explicit imports — no `from module import *`.
- Group imports: stdlib → third-party → internal, separated by blank lines.

---

## FastAPI conventions

1. All routes that return sensitive data must declare an explicit `response_model`.
2. CORS must not use `allow_origins=["*"]` on non-public routes.
3. Pydantic models on auth/sensitive routes must use `model_config = ConfigDict(extra="forbid")`.
4. Dependency injection via `Depends()` is preferred over importing shared state directly.
5. Route handlers must be thin — business logic belongs in service modules under `api/services/`.

---

## Security conventions

1. No secrets, API keys, or tokens in source code — use environment variables via `pydantic-settings`.
2. Never log passwords, tokens, or PII.
3. All user-controlled paths must be validated with `os.path.normpath` and reject `..` traversal.
4. Never use `subprocess` with `shell=True` and user-controlled input.
5. Never use `eval()` or `exec()` on user input.
6. All protected routes must verify the Keycloak Bearer token via the auth dependency.

---

## Naming conventions

- **Modules/files**: `snake_case` (e.g. `hybrid_parser.py`, `question_generator.py`)
- **Classes**: `PascalCase` (e.g. `HybridParser`, `SessionManager`)
- **Functions and variables**: `snake_case`
- **Constants**: `SCREAMING_SNAKE_CASE` (e.g. `MAX_DIFF_CHARS`, `DEFAULT_TIMEOUT`)
- **Boolean variables**: must start with `is_`, `has_`, `can_`, or `should_` (e.g. `is_valid`, `has_permission`)
- **Acronyms in names**: all uppercase (e.g. `parse_pdf`, `get_url`, `user_id`)
- No abbreviations unless widely accepted (`url`, `pdf`, `llm`, `rag`)

---

## Type hint conventions

- All public function signatures must be fully annotated.
- Use `X | None` (Python 3.10+ union syntax) instead of `Optional[X]`.
- Use `list[X]`, `dict[K, V]` (lowercase) instead of `List[X]`, `Dict[K, V]`.
- Avoid `Any` — use a specific type or `TypeVar` where generics are needed.
- Use `TypedDict` or `dataclass` for structured dicts passed between functions.

---

## Testing conventions

- Test files live in `tests/` and mirror the `api/` structure.
- Unit tests use `pytest` with `httpx.AsyncClient` for FastAPI route testing.
- Mock external services (Redis, Milvus, LiteLLM) at the boundary — do not hit real services in unit tests.
- New route handlers must have at least one happy-path and one error-path test.
