# Aggregates per-chunk Milvus search hits into per-document match reports.
# Compatibility score = aggregation of a document's chunk scores (default: max — the best
# chunk represents the doc). "mean" (legacy) and "top_k_mean" are also available; "max"
# avoids letting a pile of weak chunks drag down a doc that has one excellent match.
from collections import defaultdict
from typing import Literal

import numpy as np

from api.models import (
    InputQueryChunkMatch,
    InputQueryDocumentMatchReport,
    InputQueryDocumentMatchSummary,
    InputQueryOverallReport,
    MilvusSearchHit,
    TopicEnum,
)
from api.services.corpus_storage import CorpusStorageService

ScoreAggregation = Literal["max", "mean", "top_k_mean"]


def aggregate_doc_score(scores: list[float], aggregation: ScoreAggregation, top_k: int) -> float:
    """Reduce a document's per-chunk scores to a single compatibility score.

    "max" takes the best chunk (a strong hit is not diluted by unrelated chunks), "mean"
    is the legacy all-chunk average, and "top_k_mean" averages the strongest `top_k`
    chunks. Empty input scores 0.0.
    """
    if not scores:
        return 0.0
    if aggregation == "mean":
        return float(np.mean(scores))
    if aggregation == "top_k_mean":
        top_scores = sorted(scores, reverse=True)[: max(1, top_k)]
        return float(np.mean(top_scores))
    return float(max(scores))  # "max" (default)


def generate_doc_match_report(
    topic: TopicEnum,
    milvus_search_results: list[list[MilvusSearchHit]],
    input_queries: list[str],
    corpus_storage: CorpusStorageService,
    aggregation: ScoreAggregation = "max",
    top_k: int = 3,
) -> list[InputQueryOverallReport]:
    # Flatten + group
    final_reports: list[InputQueryOverallReport] = []

    for input_query_index, input_query_vector_results in enumerate(milvus_search_results):
        doc_chunks: dict[str, list] = defaultdict(list)  # doc_id -> list of chunk dicts
        doc_scores: dict[str, list] = defaultdict(list)  # doc_id -> list of scores
        final_docs = []

        for chunk in input_query_vector_results:
            doc_chunks[chunk.doc_id].append(chunk)  # doc_id = Minio id of the corpus chunk
            doc_scores[chunk.doc_id].append(chunk.distance)

        for doc_id, chunks in doc_chunks.items():
            # Single lookup covers both URL and filename — was two separate list_objects calls.
            obj = corpus_storage.find_object(doc_id, topic)
            if not obj:
                continue  # skip missing docs; was returning None and aborting the whole report

            doc_url = corpus_storage.presigned_url_for(doc_id, topic)
            file_name = obj.metadata.get("x-amz-meta-original-filename", doc_id)

            compatibility_score = aggregate_doc_score(doc_scores[doc_id], aggregation, top_k)

            # Convert to InputQueryChunkMatch
            chunk_matches = [
                InputQueryChunkMatch(
                    corpus_chunk=c.chunk_text,
                    corpus_page_num=c.page_num,
                    score=c.distance,
                    doc_url=doc_url,
                )
                for c in chunks
            ]

            final_docs.append(
                InputQueryDocumentMatchReport(
                    doc_id=doc_id,
                    file_name=file_name,
                    doc_url=doc_url,
                    compatibility_score=compatibility_score,
                    chunk_matches=chunk_matches,
                )
            )

        final_docs.sort(key=lambda x: x.compatibility_score, reverse=True)

        # Summary of the top 'limit' fetched documents
        input_query_report = InputQueryOverallReport(
            summary={
                "input_query": input_queries[input_query_index],
                "total_matched_docs": len(final_docs),
                "target_matching_files": [
                    InputQueryDocumentMatchSummary(
                        doc_id=doc.doc_id,
                        file_name=doc.file_name,
                        doc_url=doc.doc_url,
                        compatibility_score=doc.compatibility_score,
                    )
                    for doc in final_docs
                ],
            },
            detailed_matches=final_docs,
        )
        final_reports.append(input_query_report)

    return final_reports
