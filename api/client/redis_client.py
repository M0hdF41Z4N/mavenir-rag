import logging
import time
from datetime import UTC, datetime

from redis import Redis
from redis.exceptions import RedisError
from redisvl.extensions.message_history import MessageHistory

from api.config import RedisSettings

logger = logging.getLogger(__name__)

DEFAULT_SESSION_PREVIEW_CHARS = 120
REDIS_SOCKET_CONNECT_TIMEOUT_SECONDS = 5
REDIS_SOCKET_TIMEOUT_SECONDS = 5


class RedisClient:
    """
    Redis-backed session store with a unified key namespace.

    Key schema (all session data lives under {username}:{session_id}:*):
        {username}:{session_id}:meta           Hash  — session metadata
        {username}:{session_id}:chat:*         redisvl-managed message history
        user:{username}:sessions               ZSet  — per-user session index (recency-ordered)
    """

    def __init__(self, settings: RedisSettings):
        self.redis = Redis.from_url(
            settings.redis_url,
            decode_responses=True,
            socket_connect_timeout=REDIS_SOCKET_CONNECT_TIMEOUT_SECONDS,
            socket_timeout=REDIS_SOCKET_TIMEOUT_SECONDS,
        )

    # -------------------------
    # Internal key/handle helpers
    # -------------------------
    def _meta_key(self, username: str, session_id: str) -> str:
        return f"{username}:{session_id}:meta"

    def _index_key(self, username: str) -> str:
        return f"user:{username}:sessions"

    def _history(self, username: str, session_id: str) -> MessageHistory:
        return MessageHistory(
            name=f"{username}:{session_id}",
            session_tag="chat",
            redis_client=self.redis,
        )

    # -------------------------
    # Messages
    # -------------------------
    def store_message(
        self,
        username: str,
        session_id: str,
        prompt: str,
        response: str,
        sources: list[dict[str, object]] | None = None,
    ) -> None:
        assistant_msg: dict[str, object] = {"role": "assistant", "content": response, "metadata": {"sources": sources}}
        self._history(username, session_id).add_messages([{"role": "user", "content": prompt}, assistant_msg])

    def get_recent_messages(self, username: str, session_id: str, top_k: int = 5) -> list[dict[str, object]]:
        return self._history(username, session_id).get_recent(top_k=top_k)

    # -------------------------
    # Session metadata + index
    # -------------------------
    def get_session_meta(self, username: str, session_id: str) -> dict[str, str] | None:
        data = self.redis.hgetall(self._meta_key(username, session_id))
        return data if data else None

    def register_session(
        self,
        username: str,
        session_id: str,
        topic: str,
        first_query: str,
        is_new: bool,
    ) -> None:
        """Register or update a session's metadata and index entry.

        Called once per chat turn — after the caller has stored exactly one
        prompt + one response pair via store_message. message_count is therefore
        seeded at 2 on creation and incremented by 2 on each subsequent turn.
        """
        meta_key = self._meta_key(username, session_id)
        index_key = self._index_key(username)
        now = datetime.now(UTC).isoformat()

        # Treat resume of an unknown session as new (handles pre-feature sessions)
        if not is_new and not self.redis.exists(meta_key):
            is_new = True
            first_query = ""

        with self.redis.pipeline() as pipe:
            if is_new:
                pipe.hset(
                    meta_key,
                    mapping={
                        "topic": topic,
                        "created_at": now,
                        "last_updated": now,
                        "message_count": 2,
                        "preview": first_query[:DEFAULT_SESSION_PREVIEW_CHARS],
                    },
                )
            else:
                pipe.hset(meta_key, mapping={"last_updated": now})
                pipe.hincrby(meta_key, "message_count", 2)
            pipe.zadd(index_key, {session_id: time.time()})
            try:
                pipe.execute()
            except RedisError:
                logger.exception("Failed to register session %s for user %s", session_id, username)
                raise

    def list_user_sessions(self, username: str, start: int = 0, limit: int = 20) -> list[dict[str, str]]:
        index_key = self._index_key(username)
        session_ids = self.redis.zrevrange(index_key, start, start + limit - 1)
        if not session_ids:
            return []
        with self.redis.pipeline() as pipe:
            for sid in session_ids:
                pipe.hgetall(self._meta_key(username, sid))
            try:
                results = pipe.execute()
            except RedisError:
                logger.exception("Failed to list sessions for user %s", username)
                raise
        return [{**meta, "session_id": sid} for sid, meta in zip(session_ids, results, strict=True) if meta]

    def delete_session(self, username: str, session_id: str) -> None:
        # Clear chat history first so a failure here leaves meta + index intact,
        # keeping the session discoverable for retry instead of orphaning chat keys.
        self._history(username, session_id).clear()
        with self.redis.pipeline() as pipe:
            pipe.delete(self._meta_key(username, session_id))
            pipe.zrem(self._index_key(username), session_id)
            try:
                pipe.execute()
            except RedisError:
                logger.exception("Failed to delete session %s for user %s", session_id, username)
                raise
