import html
import json
from pathlib import Path

_STYLES = """
  :root { color-scheme: light dark; }
  * { box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         max-width: 980px; margin: 2rem auto; padding: 0 1.25rem; line-height: 1.55;
         color: #1b1f23; background: #fff; }
  h1 { border-bottom: 2px solid #0366d6; padding-bottom: .4rem; }
  h2 { margin-top: 2rem; border-bottom: 1px solid #e1e4e8; padding-bottom: .3rem; }
  h3 { margin-top: 1.6rem; }
  .meta { background: #f6f8fa; border: 1px solid #e1e4e8; border-radius: 6px;
          padding: .9rem 1rem; margin: 1rem 0 1.5rem; }
  .meta dl { display: grid; grid-template-columns: max-content 1fr; gap: .35rem 1rem; margin: 0; }
  .meta dt { font-weight: 600; color: #586069; }
  .meta dd { margin: 0; word-break: break-all; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: .92em; }
  .qa { border: 1px solid #e1e4e8; border-left: 4px solid #0366d6;
        border-radius: 6px; padding: .8rem 1rem; margin-bottom: 1rem; background: #fafbfc; }
  .qa .q { font-weight: 600; color: #0366d6; }
  .qa .a { margin-top: .4rem; white-space: pre-wrap; }
  .qa .src { margin-top: .4rem; color: #586069; font-size: .9em; }
  table { border-collapse: collapse; width: 100%; margin: .5rem 0 1rem; font-size: .92em; }
  th, td { border: 1px solid #e1e4e8; padding: .4rem .6rem; text-align: left; vertical-align: top; }
  th { background: #f6f8fa; }
  code, pre { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
  pre { background: #f6f8fa; border: 1px solid #e1e4e8; border-radius: 6px;
        padding: .8rem 1rem; overflow: auto; font-size: .85em; max-height: 500px; }
  details { margin-top: 1.2rem; }
  summary { cursor: pointer; font-weight: 600; color: #0366d6; }
  a { color: #0366d6; }
  .answer { white-space: pre-wrap; background: #fff; border: 1px solid #e1e4e8;
            border-radius: 6px; padding: .8rem 1rem; }
  ol.qlist li { margin: .2rem 0; }
"""


def _esc(value: object) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _meta_html(meta: dict) -> str:
    rows = [
        ("Run timestamp", meta.get("timestamp", "")),
        ("Task ID", meta.get("task_id", "")),
        ("Doc ID", meta.get("doc_id", "")),
        ("Topic", meta.get("topic", "")),
        ("Session ID", meta.get("session_id", "")),
        ("Ingestion elapsed", meta.get("ingestion_elapsed", "n/a")),
        ("Sample PDF", meta.get("sample_pdf", "")),
    ]
    items = "\n".join(f"      <dt>{_esc(k)}</dt><dd>{_esc(v)}</dd>" for k, v in rows)
    return f'<section class="meta">\n    <dl>\n{items}\n    </dl>\n  </section>'


def _doc_shell(title: str, body: str) -> str:
    return (
        "<!doctype html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '  <meta charset="utf-8">\n'
        f"  <title>{_esc(title)}</title>\n"
        f"  <style>{_STYLES}</style>\n"
        "</head>\n"
        "<body>\n"
        f"  <h1>{_esc(title)}</h1>\n"
        f"{body}\n"
        "</body>\n"
        "</html>\n"
    )


def write_summary(run_dir: Path, meta: dict, qa_pairs: list[tuple[str, dict]]) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "summary.html"

    qa_html_parts: list[str] = []
    for i, (question, response) in enumerate(qa_pairs, start=1):
        answer = (response.get("answer") or "").strip()
        excerpt = answer[:250] + ("…" if len(answer) > 250 else "")
        src_count = len(response.get("sources") or [])
        qa_html_parts.append(
            f'  <div class="qa">\n    <div class="q">Q{i}. {_esc(question)}</div>\n    <div class="a">{_esc(excerpt)}</div>\n    <div class="src">Sources: {src_count}</div>\n  </div>'
        )

    body = f'  {_meta_html(meta)}\n  <p><strong>Questions asked:</strong> {len(qa_pairs)}</p>\n  <section class="qa-list">\n' + "\n".join(qa_html_parts) + "\n  </section>"
    path.write_text(_doc_shell("E2E RAG Smoke Test — Summary", body), encoding="utf-8")
    return path.resolve()


def _sources_table(sources: list[dict]) -> str:
    if not sources:
        return "<p><em>No sources returned.</em></p>"
    rows: list[str] = []
    for i, s in enumerate(sources, start=1):
        excerpt = (s.get("chunk_excerpt") or "")[:240]
        doc_url = s.get("doc_url")
        doc_link = f'<a href="{_esc(doc_url)}" target="_blank" rel="noopener">{_esc(s.get("doc_id", ""))}</a>' if doc_url else _esc(s.get("doc_id", ""))
        rows.append(
            "      <tr>"
            f"<td>{i}</td>"
            f"<td>{doc_link}</td>"
            f"<td>{_esc(s.get('file_name', ''))}</td>"
            f"<td>{_esc(s.get('page_num', ''))}</td>"
            f"<td>{_esc(s.get('relevance_score', ''))}</td>"
            f"<td>{_esc(excerpt)}</td>"
            "</tr>"
        )
    return (
        "    <table>\n"
        "      <thead><tr>"
        "<th>#</th><th>doc_id</th><th>file</th><th>page</th><th>score</th><th>excerpt</th>"
        "</tr></thead>\n"
        "      <tbody>\n" + "\n".join(rows) + "\n      </tbody>\n"
        "    </table>"
    )


def write_detailed(
    run_dir: Path,
    meta: dict,
    infer_response: dict,
    qa_pairs: list[tuple[str, dict]],
) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "detailed.html"

    infer_result = (infer_response or {}).get("result") or {}
    report_url = infer_result.get("markdown_report_url")
    report_url_block = f'  <p><strong>Infer markdown report URL:</strong> <a href="{_esc(report_url)}" target="_blank" rel="noopener">{_esc(report_url)}</a></p>\n' if report_url else ""

    qlist = "\n".join(f"    <li>{_esc(q)}</li>" for q, _ in qa_pairs) or "    <li><em>None</em></li>"

    transcripts: list[str] = []
    for i, (question, response) in enumerate(qa_pairs, start=1):
        answer = response.get("answer") or "(empty)"
        transcripts.append(
            f"  <h3>Q{i}</h3>\n"
            f"  <p><strong>Question:</strong> {_esc(question)}</p>\n"
            f"  <p><strong>Answer:</strong></p>\n"
            f'  <div class="answer">{_esc(answer)}</div>\n'
            f"  <p><strong>Sources:</strong></p>\n"
            f"{_sources_table(response.get('sources') or [])}"
        )

    try:
        pretty = json.dumps(infer_response, indent=2, default=str)
    except Exception as e:
        pretty = f"<failed to serialize infer response: {e}>"
    if len(pretty) > 20000:
        pretty = pretty[:20000] + "\n... [truncated]"

    body = (
        f"  {_meta_html(meta)}\n"
        f"{report_url_block}"
        "  <h2>Extracted Questions</h2>\n"
        '  <ol class="qlist">\n' + qlist + "\n  </ol>\n"
        "  <h2>Chat Transcripts</h2>\n" + "\n".join(transcripts) + "\n"
        "  <details>\n"
        "    <summary>Raw Infer Response (truncated)</summary>\n"
        f"    <pre>{_esc(pretty)}</pre>\n"
        "  </details>"
    )
    path.write_text(_doc_shell("E2E RAG Smoke Test — Detailed Report", body), encoding="utf-8")
    return path.resolve()
