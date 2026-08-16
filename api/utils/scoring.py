# Aggregates per-chunk Milvus search hits into per-document match reports.
# Compatibility score = arithmetic mean of all chunk cosine distances for a document.
# Note: mean averaging treats all chunks equally — a doc with one very strong match
# and many weak ones scores the same as a doc with uniformly moderate relevance.
from collections import defaultdict

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


def generate_doc_match_report(
    topic: TopicEnum,
    milvus_search_results: list[list[MilvusSearchHit]],
    input_queries: list[str],
    corpus_storage: CorpusStorageService,
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

            # Average of ALL chunk scores
            doc_url = corpus_storage.presigned_url_for(doc_id, topic)
            file_name = obj.metadata.get("x-amz-meta-original-filename", doc_id)

            compatibility_score = float(np.mean(doc_scores[doc_id]))  # average of ALL chunk scores

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
