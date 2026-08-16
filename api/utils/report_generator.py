# Generates Markdown match reports from inference pipeline results and uploads them to MinIO.
# The report is written to a local file first, then uploaded — the local file is NOT deleted
# after upload (known limitation: see file-handling.md). Each report gets a unique UUID in
# its MinIO path to prevent concurrent inference calls from overwriting each other's file.
import logging
import os
import uuid

from api.client.minio_client import MinioClient
from api.models import InputQueryDocumentMatchReport, InputQueryOverallReport

logger = logging.getLogger(__name__)


def generate_summarized_table(milvus_reports: InputQueryOverallReport, markdown_code: str) -> str:
    """
    Generate a Markdown table index for a list of milvus reports,
    with separate columns for source/target file names and doc IDs.
    """

    markdown_code.append("## Report Summary\n")

    # ---------- Table Header ----------
    markdown_code.append("| Input Query No. | Input Query | Total Matched Docs | Target File Name | Target Doc ID | Target Doc URL | Compatibility Score |")
    markdown_code.append("|---|---|---|---|---|---|---|")

    for idx, report in enumerate(milvus_reports, start=1):
        summary = report.summary
        total_matched_docs = summary.total_matched_docs

        # ---------- No matches ----------
        if not summary.target_matching_files:
            markdown_code.append(f"| {idx} | {summary.input_query} | {total_matched_docs} | - | - | - | - |")
            continue

        # ---------- First row ----------
        first_file = summary.target_matching_files[0]
        markdown_code.append(
            f"| {idx} | {summary.input_query} | {total_matched_docs} | {first_file.file_name} | {first_file.doc_id} | {first_file.doc_url} | {first_file.compatibility_score:.4f} |"
        )

        # ---------- Remaining rows ----------
        for f in summary.target_matching_files[1:]:
            markdown_code.append(f"|   |   |   | {f.file_name} | {f.doc_id} | {f.doc_url} | {f.compatibility_score:.4f} |")

    markdown_code.append("\n")
    return markdown_code


def generate_detailed_matches_for_input_query(
    input_query: str,
    detailed_matches: list[InputQueryDocumentMatchReport],
    markdown_code: str,
) -> str:
    """
    Generate a flattened Markdown table of detailed chunk matches for the given input query.

    Columns:
    Input Query |
    Corpus File Name | Corpus Doc ID | Corpus Chunk |
    Chunk Score | Compatibility Score
    """

    markdown_code.append("## Detailed Matches\n")

    # ---------- Table Header ----------
    markdown_code.append("| Input Query | Corpus File Name | Corpus Doc ID | Corpus Chunk | Corpus Page | LLM Questions | Chunk Score | Compatibility Score |")

    markdown_code.append("|---|---|---|---|---|---|---|---|")

    # ---------- Rows ----------
    for match in detailed_matches:
        corpus_file_name = match.file_name
        corpus_doc_id = match.doc_id
        compatibility_score = match.compatibility_score

        for chunk in match.chunk_matches:
            input_query = input_query.replace("\n", " ")
            corpus_chunk = chunk.corpus_chunk.replace("\n", " ")
            corpus_page = chunk.corpus_page_num
            chunk_score = chunk.score
            llm_resp = chunk.llm_generated_questions
            if llm_resp and llm_resp.questions:
                questions = " ".join(f"Q{i + 1}: {qa.question} A: {qa.answer}" for i, qa in enumerate(llm_resp.questions))
            elif llm_resp and llm_resp.match_sets:
                questions = " ".join(f"{ms.set_label}: {ms.answer}" for ms in llm_resp.match_sets)
            elif llm_resp and llm_resp.raw:
                questions = llm_resp.raw.replace("\n", " ")
            else:
                questions = ""

            markdown_code.append(f"| {input_query} | {corpus_file_name} | {corpus_doc_id} | {corpus_chunk} | {corpus_page} | {questions} | {chunk_score:.4f} | {compatibility_score:.4f} |")

    markdown_code.append("\n")

    return markdown_code


def generate_markdown_report(
    milvus_reports: list[InputQueryOverallReport],
    output_md_file_path: str,
    minio_client: MinioClient,
) -> str:
    """
    Generate a Markdown (.md) report from a list of `generate_doc_match_report` outputs.

    Args:
        doc_match_reports (list[dict]): List of reports returned by `generate_doc_match_report`.
        output_md_file_path (str): Path to save the combined Markdown report.
    """
    lines = []

    lines.append("# Document Match Report\n")

    lines = generate_summarized_table(milvus_reports, lines)

    for idx, report in enumerate(milvus_reports, start=1):
        input_query = report.summary.input_query
        lines.append("\n---\n")  # section breaker
        lines.append(f"## Input Query {idx}: {input_query}\n")
        detailed_matches = report.detailed_matches
        lines = generate_detailed_matches_for_input_query(
            input_query=input_query,
            detailed_matches=detailed_matches,
            markdown_code=lines,
        )

    # Write to file
    with open(output_md_file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # Upload to MinIO
    filename = os.path.basename(output_md_file_path)
    minio_report_id = str(uuid.uuid4())
    object_name = f"generated_reports/{minio_report_id}/{filename}"

    try:
        minio_client.upload_file(
            file_path=output_md_file_path,
            object_name=object_name,
        )
    except Exception as e:
        logger.error(f"Failed to upload report to MinIO: {e}", exc_info=True)

    markdown_report_url = minio_client.presigned_get_url(object_name)
    return markdown_report_url
