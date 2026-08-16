from api.utils.parsers.base import DocumentParser
from api.utils.parsers.hybrid_parser import process_document_for_tasks


def get_parser(settings) -> DocumentParser:
    """Return the appropriate document parser based on configuration.

    Currently always returns the hybrid parser (pymupdf4llm + docling).
    The settings parameter is accepted for forward-compatibility — a future
    lightweight parser could be returned when use_docling_hybrid=False.
    """
    return process_document_for_tasks


__all__ = [
    "DocumentParser",
    "get_parser",
    "process_document_for_tasks",
]
