from typing import Any


def _coerce_to_questions(raw: Any) -> list[str]:
    """Normalize one `llm_generated_questions` payload into a flat list of question strings."""
    if raw is None:
        return []
    if isinstance(raw, dict):
        questions_field = raw.get("questions")
        if isinstance(questions_field, list):
            out: list[str] = []
            for item in questions_field:
                if isinstance(item, dict) and isinstance(item.get("question"), str):
                    out.append(item["question"])
                elif isinstance(item, str):
                    out.append(item)
            if out:
                return out
        raw_str = raw.get("raw")
        if isinstance(raw_str, str):
            return [line.strip("-• ").strip() for line in raw_str.splitlines() if line.strip()]
        return []
    if isinstance(raw, str):
        return [line.strip("-• ").strip() for line in raw.splitlines() if line.strip()]
    return []


def extract_questions(infer_response: dict, limit: int) -> list[str]:
    """Walk the infer response and return up to `limit` unique questions.

    Path: result.results[*].detailed_matches[*].chunk_matches[*].llm_generated_questions
    """
    result_block = (infer_response or {}).get("result") or {}
    overall_reports = result_block.get("results") or []

    collected: list[str] = []
    for report in overall_reports:
        for match in report.get("detailed_matches") or []:
            for chunk in match.get("chunk_matches") or []:
                payload = chunk.get("llm_generated_questions")
                collected.extend(_coerce_to_questions(payload))

    seen: set[str] = set()
    unique: list[str] = []
    for q in collected:
        q = q.strip()
        if len(q) < 5:
            continue
        key = q.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(q)
        if len(unique) >= limit:
            break

    if not unique:
        raise RuntimeError("No questions generated — infer response did not contain usable questions")
    return unique
