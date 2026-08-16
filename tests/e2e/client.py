from pathlib import Path

import httpx

from tests.e2e.config import BASE_URL, HTTP_TIMEOUT_SEC


class ApiError(RuntimeError):
    pass


class E2EClient:
    def __init__(self, base_url: str = BASE_URL, timeout: float = HTTP_TIMEOUT_SEC):
        self._client = httpx.Client(base_url=base_url, timeout=timeout)

    def close(self) -> None:
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def health(self) -> bool:
        try:
            r = self._client.get("/docs")
            return r.status_code == 200
        except httpx.HTTPError:
            return False

    def ingest_file(self, pdf_path: Path, topic: str) -> dict:
        with pdf_path.open("rb") as fh:
            files = {"file": (pdf_path.name, fh, "application/pdf")}
            data = {"topic": topic}
            r = self._client.post("/corpus/document/ingest-file", files=files, data=data)
        self._raise(r, "ingest_file")
        return r.json()

    def get_task(self, task_id: str) -> dict:
        r = self._client.get(f"/corpus/task/{task_id}")
        self._raise(r, "get_task")
        return r.json()

    def infer(self, doc_id: str, topic: str, session_id: str) -> dict:
        params = [
            ("topic", topic),
            ("question_answer_type", "subjective"),
            ("knowledge_feed_mode", "text"),
            ("input_queries", doc_id),
            ("session_id", session_id),
        ]
        r = self._client.post("/rigil/document/infer", params=params)
        self._raise(r, "infer")
        return r.json()

    def chat(self, query: str, topic: str, session_id: str) -> dict:
        params = {"query": query, "topic": topic, "session_id": session_id}
        r = self._client.post("/rigil/document/chat", params=params)
        self._raise(r, "chat")
        return r.json()

    @staticmethod
    def _raise(response: httpx.Response, op: str) -> None:
        if response.is_success:
            return
        body = response.text[:500]
        raise ApiError(f"{op} failed: HTTP {response.status_code} — {body}")
