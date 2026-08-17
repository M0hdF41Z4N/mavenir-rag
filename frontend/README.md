# Streamlit Chat Frontend

A minimal Streamlit UI wired to the existing backend chat endpoint
(`POST /rigil/document/chat`). It runs as a **separate process** and talks to the
FastAPI backend over HTTP — no backend code changes are required.

## Features

- Corpus-grounded chat with per-answer source citations (file, page, link, score).
- Topic selector (matches the backend `TopicEnum`).
- Session sidebar: list recent conversations, load one, start a new chat, delete.
- Session continuity via the backend's Redis-backed `session_id`.

## Prerequisites

1. Install dependencies (Streamlit is in the dev group):

   ```bash
   poetry install
   ```

2. Start the backend (and its Redis / Milvus / MinIO dependencies) on port `8001`.
   For local use, set `APP_ENV=dev` so the backend **skips Keycloak auth** and
   uses a fixed test user — the UI then needs no token:

   ```bash
   poetry run serve      # or: poetry run start  (with reload)
   ```

## Run the UI

From the repo root:

```bash
poetry run poe ui
# equivalent to:
poetry run streamlit run frontend/streamlit_app.py
```

The app opens at http://localhost:8501.

## Configuration

The frontend reads two environment variables:

| Variable       | Default                 | Purpose                                                                 |
| -------------- | ----------------------- | ----------------------------------------------------------------------- |
| `API_BASE_URL` | `http://localhost:8001` | Backend base URL (the `/rigil` prefix is added automatically).          |
| `API_TOKEN`    | _(unset)_               | Keycloak access token. **Leave unset in `APP_ENV=dev`.** Required for any non-dev deployment; sent as `Authorization: Bearer <token>`. |

Example against a non-dev backend:

```bash
API_BASE_URL=https://rag.example.com API_TOKEN="<access_token>" \
  poetry run streamlit run frontend/streamlit_app.py
```

Obtain a token via `POST /rigil/auth/token` (Keycloak refresh-token exchange).

## Files

- `api_client.py` — typed synchronous HTTP wrapper around the chat + session endpoints.
- `streamlit_app.py` — the UI (chat area, source rendering, session sidebar).
