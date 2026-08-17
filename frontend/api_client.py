"""Thin synchronous HTTP client for the Rigil chat API.

The Streamlit frontend runs as a separate process and talks to the FastAPI
backend over HTTP. This module wraps the four endpoints the UI needs and keeps
the frontend decoupled from the backend's Pydantic models (it returns plain
dicts / lightweight dataclasses).

Configuration comes from the environment:
  - ``API_BASE_URL`` — backend base URL (default ``http://localhost:8001``)
  - ``API_TOKEN``    — optional Keycloak access token. When ``APP_ENV=dev`` the
                       backend skips auth, so this can be left unset for local
                       development. Set it for any non-dev deployment.
"""

import logging
import os
from dataclasses import dataclass, field

import httpx

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://localhost:8001"
API_PREFIX = "/rigil"
DEFAULT_TIMEOUT_SECONDS = 60.0

# Topics accepted by the backend TopicEnum. Kept as a plain tuple so the
# frontend has no import-time dependency on the backend package.
TOPICS: tuple[str, ...] = (
    "general",
    "aviation",
    "finance",
    "healthcare",
    "technology",
    "legal",
    "hr",
)


class ApiError(Exception):
    """Raised when the backend returns an error or is unreachable."""


@dataclass
class ChatSource:
    doc_id: str
    file_name: str
    doc_url: str
    relevance_score: float
    chunk_excerpt: str
    page_num: int | None = None


@dataclass
class ChatMessage:
    role: str
    content: str
    sources: list[ChatSource] = field(default_factory=list)


@dataclass
class ChatResult:
    answer: str
    session_id: str | None
    sources: list[ChatSource]
    history: list[ChatMessage]


@dataclass
class SessionSummary:
    session_id: str
    topic: str
    created_at: str
    last_updated: str
    message_count: int
    preview: str


def _base_url() -> str:
    return os.getenv("API_BASE_URL", DEFAULT_BASE_URL).rstrip("/")


def _auth_headers() -> dict[str, str]:
    token = os.getenv("API_TOKEN")
    if token:
        return {"Authorization": f"Bearer {token}"}
    return {}


def _parse_source(raw: dict[str, object]) -> ChatSource:
    return ChatSource(
        doc_id=str(raw.get("doc_id", "")),
        file_name=str(raw.get("file_name", "")),
        doc_url=str(raw.get("doc_url", "")),
        relevance_score=float(raw.get("relevance_score", 0.0)),
        chunk_excerpt=str(raw.get("chunk_excerpt", "")),
        page_num=raw.get("page_num"),  # type: ignore[arg-type]
    )


def _parse_message(raw: dict[str, object]) -> ChatMessage:
    raw_sources = raw.get("sources") or []
    sources = [_parse_source(s) for s in raw_sources]  # type: ignore[union-attr]
    return ChatMessage(
        role=str(raw.get("role", "assistant")),
        content=str(raw.get("content", "")),
        sources=sources,
    )


def _request(method: str, path: str, *, params: dict[str, object]) -> object:
    """Perform an HTTP request against the backend and return parsed JSON.

    Uses a context-managed client so the connection is always closed. All
    transport and HTTP-status failures are converted to ``ApiError`` with a
    caller-friendly message and logged with ``logger.exception``.
    """
    url = f"{_base_url()}{API_PREFIX}{path}"
    try:
        with httpx.Client(timeout=DEFAULT_TIMEOUT_SECONDS, headers=_auth_headers()) as client:
            response = client.request(method, url, params=params)
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as exc:
        detail = _extract_detail(exc.response)
        logger.exception("Backend returned an error for %s %s", method, path)
        raise ApiError(detail) from exc
    except httpx.HTTPError as exc:
        logger.exception("Failed to reach backend for %s %s", method, path)
        raise ApiError(f"Could not reach the API at {_base_url()}: {exc}") from exc


def _extract_detail(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return f"HTTP {response.status_code}: {response.text[:200]}"
    detail = payload.get("detail") if isinstance(payload, dict) else None
    return str(detail) if detail else f"HTTP {response.status_code}"


def send_chat(query: str, topic: str, session_id: str | None) -> ChatResult:
    """Send a user query to the chat endpoint and return the grounded answer."""
    params: dict[str, object] = {"query": query, "topic": topic}
    if session_id:
        params["session_id"] = session_id

    data = _request("POST", "/document/chat", params=params)
    if not isinstance(data, dict):
        raise ApiError("Unexpected response shape from chat endpoint.")

    return ChatResult(
        answer=str(data.get("answer", "")),
        session_id=data.get("session_id"),  # type: ignore[arg-type]
        sources=[_parse_source(s) for s in (data.get("sources") or [])],  # type: ignore[union-attr]
        history=[_parse_message(m) for m in (data.get("history") or [])],  # type: ignore[union-attr]
    )


def list_sessions() -> list[SessionSummary]:
    """List the current user's recent chat sessions, newest first."""
    data = _request("GET", "/document/sessions", params={})
    if not isinstance(data, dict):
        return []
    sessions = data.get("sessions") or []
    return [
        SessionSummary(
            session_id=str(s.get("session_id", "")),
            topic=str(s.get("topic", "")),
            created_at=str(s.get("created_at", "")),
            last_updated=str(s.get("last_updated", "")),
            message_count=int(s.get("message_count", 0)),
            preview=str(s.get("preview", "")),
        )
        for s in sessions  # type: ignore[union-attr]
    ]


def get_session(session_id: str) -> list[ChatMessage]:
    """Fetch the full message history for a session to rehydrate the UI."""
    data = _request("GET", f"/document/sessions/{session_id}", params={})
    if not isinstance(data, dict):
        return []
    return [_parse_message(m) for m in (data.get("history") or [])]  # type: ignore[union-attr]


def delete_session(session_id: str) -> None:
    """Clear a conversation session from the backend."""
    _request("DELETE", "/document/session", params={"session_id": session_id})
