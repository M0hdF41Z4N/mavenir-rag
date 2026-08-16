import sys
import time
import uuid
from datetime import datetime

from tests.e2e.client import ApiError, E2EClient
from tests.e2e.config import (
    BASE_URL,
    CHAT_QUESTION_LIMIT,
    POLL_INTERVAL_SEC,
    POLL_TIMEOUT_SEC,
    REPORTS_DIR,
    SAMPLE_PDF,
    TOPIC,
)
from tests.e2e.question_extractor import extract_questions
from tests.e2e.report_writer import write_detailed, write_summary


def _log(msg: str) -> None:
    print(msg, flush=True)


def _poll_until_done(client: E2EClient, task_id: str) -> dict:
    start = time.time()
    attempt = 0
    while True:
        task = client.get_task(task_id)
        status = task.get("status")
        elapsed = int(time.time() - start)
        progress = task.get("progress") or ""
        _log(f"[poll #{attempt} t={elapsed}s] status={status} progress={progress}")
        if status == "completed":
            return task
        if status == "failed":
            raise ApiError(f"Task {task_id} failed: {task.get('error')}")
        if elapsed >= POLL_TIMEOUT_SEC:
            raise TimeoutError(f"Task {task_id} did not complete within {POLL_TIMEOUT_SEC}s (last status={status})")
        attempt += 1
        time.sleep(POLL_INTERVAL_SEC)


def main() -> int:
    _log("=" * 60)
    _log("E2E RAG Smoke Test")
    _log("=" * 60)

    if not SAMPLE_PDF.exists():
        _log(f"ERROR: sample PDF not found at {SAMPLE_PDF}")
        return 1

    with E2EClient() as client:
        if not client.health():
            _log(f"ERROR: API not reachable at {BASE_URL}")
            _log("Start it with: poetry run setup:all && poetry run start")
            return 1
        _log(f"API reachable at {BASE_URL}")

        try:
            _log(f"Ingesting {SAMPLE_PDF.name} (topic={TOPIC})...")
            ingest_resp = client.ingest_file(SAMPLE_PDF, TOPIC)
            task_id = ingest_resp.get("task_id")
            if not task_id:
                _log(f"ERROR: no task_id returned. Response: {ingest_resp}")
                return 2
            _log(f"task_id={task_id}")

            _log(f"Polling task every {POLL_INTERVAL_SEC}s (timeout {POLL_TIMEOUT_SEC}s)...")
            task = _poll_until_done(client, task_id)
            ingestion_elapsed = task.get("elapsed_time") or "n/a"
            result = task.get("result") or {}
            doc_ids = result.get("doc_ids") or []
            if not doc_ids:
                _log(f"ERROR: task completed but no doc_ids in result: {result}")
                return 2
            doc_id = doc_ids[0]
            _log(f"Task completed. doc_id={doc_id} (ingestion_elapsed={ingestion_elapsed})")

            session_id = str(uuid.uuid4())
            _log(f"Running /rigil/document/infer (session_id={session_id})...")
            infer_resp = client.infer(doc_id=doc_id, topic=TOPIC, session_id=session_id)
            markdown_url = (infer_resp.get("result") or {}).get("markdown_report_url")
            if markdown_url:
                _log(f"Infer markdown report URL: {markdown_url}")

            questions = extract_questions(infer_resp, CHAT_QUESTION_LIMIT)
            _log(f"Extracted {len(questions)} question(s) (cap={CHAT_QUESTION_LIMIT})")

            qa_pairs: list[tuple[str, dict]] = []
            for i, question in enumerate(questions, start=1):
                _log(f"[chat {i}/{len(questions)}] Q: {question[:80]}")
                chat_resp = client.chat(query=question, topic=TOPIC, session_id=session_id)
                ans_len = len(chat_resp.get("answer") or "")
                src_count = len(chat_resp.get("sources") or [])
                _log(f"    → answer_len={ans_len} sources={src_count}")
                qa_pairs.append((question, chat_resp))

            timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
            run_dir = REPORTS_DIR / timestamp
            meta = {
                "timestamp": timestamp,
                "task_id": task_id,
                "doc_id": doc_id,
                "topic": TOPIC,
                "session_id": session_id,
                "ingestion_elapsed": ingestion_elapsed,
                "sample_pdf": str(SAMPLE_PDF),
            }
            summary_path = write_summary(run_dir, meta, qa_pairs)
            detailed_path = write_detailed(run_dir, meta, infer_resp, qa_pairs)

            _log("")
            _log("Reports written:")
            _log(f"  Summary:  file://{summary_path}")
            _log(f"  Detailed: file://{detailed_path}")
            return 0

        except ApiError as e:
            _log(f"API ERROR: {e}")
            return 2
        except TimeoutError as e:
            _log(f"TIMEOUT: {e}")
            return 3
        except Exception as e:
            _log(f"UNEXPECTED ERROR: {type(e).__name__}: {e}")
            return 4


if __name__ == "__main__":
    sys.exit(main())
