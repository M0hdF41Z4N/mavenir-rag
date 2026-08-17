"""Streamlit chat frontend for the Mavenir Rag corpus-grounded RAG chatbot.

Wired to the existing ``POST /mavenir-rag/document/chat`` endpoint. Answers are
generated strictly from the ingested corpus; each answer shows its source
citations. A sidebar lists prior sessions and supports loading, starting, and
deleting conversations.

Run (from the repo root, with the backend already serving):
    streamlit run frontend/streamlit_app.py
or:
    poetry run poe ui
"""

import logging

import streamlit as st

import api_client
from api_client import ApiError, ChatMessage, ChatSource

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PAGE_TITLE = "Mavenir Rag Corpus Chat"


def _init_state() -> None:
    """Seed session_state keys on first run of a browser session."""
    st.session_state.setdefault("session_id", None)
    st.session_state.setdefault("topic", api_client.TOPICS[0])
    st.session_state.setdefault("messages", [])  # list[ChatMessage]


def _render_sources(sources: list[ChatSource]) -> None:
    if not sources:
        return
    with st.expander(f"📎 {len(sources)} source(s)"):
        for src in sources:
            page = f" · p.{src.page_num}" if src.page_num is not None else ""
            score = f" · score {src.relevance_score:.3f}"
            header = f"**{src.file_name}**{page}{score}"
            if src.doc_url:
                header += f" · [open]({src.doc_url})"
            st.markdown(header)
            if src.chunk_excerpt:
                st.caption(src.chunk_excerpt)


def _render_history() -> None:
    for message in st.session_state.messages:
        with st.chat_message(message.role):
            st.markdown(message.content)
            if message.role == "assistant":
                _render_sources(message.sources)


def _load_session(session_id: str) -> None:
    """Rehydrate the chat area from a stored session's history."""
    try:
        history = api_client.get_session(session_id)
    except ApiError as exc:
        st.error(f"Could not load session: {exc}")
        return
    st.session_state.session_id = session_id
    st.session_state.messages = history


def _start_new_session() -> None:
    st.session_state.session_id = None
    st.session_state.messages = []


def _render_sidebar() -> None:
    with st.sidebar:
        st.header("Conversations")

        st.selectbox(
            "Topic",
            options=api_client.TOPICS,
            key="topic",
            help="Restricts corpus retrieval to the selected topic.",
        )

        if st.button("➕ New chat", use_container_width=True):
            _start_new_session()
            st.rerun()

        st.divider()

        try:
            sessions = api_client.list_sessions()
        except ApiError as exc:
            st.error(f"Could not list sessions: {exc}")
            return

        if not sessions:
            st.caption("No previous sessions yet.")
            return

        for summary in sessions:
            is_active = summary.session_id == st.session_state.session_id
            label = summary.preview or summary.session_id[:8]
            cols = st.columns([5, 1])
            if cols[0].button(
                ("• " if is_active else "") + label,
                key=f"load_{summary.session_id}",
                use_container_width=True,
                help=f"{summary.topic} · {summary.message_count} messages",
            ):
                _load_session(summary.session_id)
                st.rerun()
            if cols[1].button("🗑", key=f"del_{summary.session_id}", help="Delete session"):
                _delete_session(summary.session_id)
                st.rerun()


def _delete_session(session_id: str) -> None:
    try:
        api_client.delete_session(session_id)
    except ApiError as exc:
        st.error(f"Could not delete session: {exc}")
        return
    if session_id == st.session_state.session_id:
        _start_new_session()


def _handle_query(query: str) -> None:
    """Send the query to the backend and append both turns to the transcript."""
    st.session_state.messages.append(ChatMessage(role="user", content=query))
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Searching the corpus…"):
            try:
                result = api_client.send_chat(
                    query=query,
                    topic=st.session_state.topic,
                    session_id=st.session_state.session_id,
                )
            except ApiError as exc:
                st.error(str(exc))
                # Drop the unanswered user turn so a retry starts clean.
                st.session_state.messages.pop()
                return

        st.session_state.session_id = result.session_id
        st.markdown(result.answer)
        _render_sources(result.sources)

    st.session_state.messages.append(
        ChatMessage(role="assistant", content=result.answer, sources=result.sources)
    )


def main() -> None:
    st.set_page_config(page_title=PAGE_TITLE, page_icon="💬", layout="centered")
    _init_state()
    st.title(PAGE_TITLE)
    st.caption("Answers are grounded strictly in the ingested corpus.")

    _render_sidebar()
    _render_history()

    query = st.chat_input("Ask a question about your documents…")
    if query:
        _handle_query(query)


if __name__ == "__main__":
    main()
