from pathlib import Path

BASE_URL = "http://localhost:8001"
TOPIC = "general"
POLL_INTERVAL_SEC = 20
POLL_TIMEOUT_SEC = 900
CHAT_QUESTION_LIMIT = 5
HTTP_TIMEOUT_SEC = 120.0

TESTS_DIR = Path(__file__).resolve().parents[1]
SAMPLE_PDF = TESTS_DIR / "sample_docs" / "sample.pdf"
REPORTS_DIR = TESTS_DIR / "reports"
