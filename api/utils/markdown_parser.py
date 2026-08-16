"""
Markdown Parser
===============
Splits a Markdown string into chunks based on header hierarchy.
Each chunk carries its parent heading breadcrumb as a list.
Oversized chunks are further split by token count.
"""

import re

import tiktoken

from api.config import DocParserSettings
from api.models import MarkdownParsedChunk, MarkdownParsedPage

_HEADER_RE = re.compile(r"^(#{1,6})\s+(.*)")


class MarkdownParser:
    def __init__(self, parser_settings: DocParserSettings):
        """
        Initialize with Pydantic settings.
        """
        self.chunk_max_tokens: int = parser_settings.chunk_max_tokens
        self.chunk_overlap_tokens: int = parser_settings.chunk_overlap_tokens
        self.include_header_in_chunk_content: bool = parser_settings.include_header_in_chunk_content
        self.tokenizer = tiktoken.get_encoding(parser_settings.tiktoken_encoder)

    # ------------------------------------------------------------------
    # Public Methods
    # ------------------------------------------------------------------

    def chunk(self, md_text: str) -> list[MarkdownParsedChunk]:
        """
        Parse *md_text* and return a list of Chunk objects.

        Each Chunk has:
          - content: the text under that header section
          - parent_headings: ordered list of ancestor heading texts (e.g. ["H1 text", "H2 text"])
        """
        raw_chunks = self._split_by_headers(md_text)
        return self._split_oversized(raw_chunks)

    def chunk_pdf_pages(self, pages: list[MarkdownParsedPage]) -> list[MarkdownParsedChunk]:
        """
        Accepts page-wise markdown and returns final chunks
        with page numbers preserved.
        """
        all_chunks: list[MarkdownParsedChunk] = []

        for page in pages:
            page_num = page.metadata["page"]
            md_text = page.text

            page_chunks = self.chunk(md_text)

            # attach page number to each chunk
            for chunk in page_chunks:
                chunk.page_num = page_num
                all_chunks.append(chunk)

        return all_chunks

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _split_by_headers(self, md_text: str) -> list[MarkdownParsedChunk]:
        """Single-pass split on Markdown headers."""
        chunks: list[MarkdownParsedChunk] = []

        # heading_stack[i] = text of the heading at level (i+1)
        heading_stack: list[str | None] = [None] * 6
        current_content_lines: list[str] = []
        current_headings: list[str] = []

        def _flush() -> None:
            content = "\n".join(current_content_lines).strip()
            if content or current_headings:
                chunks.append(
                    MarkdownParsedChunk(
                        content=content,
                        parent_headings=list(current_headings),
                    )
                )

        for line in md_text.splitlines():
            match = _HEADER_RE.match(line)
            if match:
                _flush()
                current_content_lines = []

                level = len(match.group(1))  # 1-6
                heading_text = match.group(2).strip()
                # Clean markdown artifacts
                heading_text = re.sub(r"\\", "", heading_text)
                heading_text = re.sub(r"\[¶\]\(.*?\)", "", heading_text).strip()

                # Update stack: set current level, clear deeper levels
                heading_stack[level - 1] = heading_text
                for i in range(level, 6):
                    heading_stack[i] = None

                # Build ordered breadcrumb (skip None slots)
                current_headings = [h for h in heading_stack if h is not None]

                if self.include_header_in_chunk_content:
                    current_content_lines.append(heading_text)
            else:
                current_content_lines.append(line)

        _flush()
        return chunks

    def _split_oversized(self, chunks: list[MarkdownParsedChunk]) -> list[MarkdownParsedChunk]:
        """Split any chunk that exceeds chunk_max_tokens into smaller pieces."""
        result: list[MarkdownParsedChunk] = []
        for chunk in chunks:
            token_count = len(self.tokenizer.encode(chunk.content))
            if token_count <= self.chunk_max_tokens:
                result.append(chunk)
            else:
                for sub_text in self._split_text_by_tokens(chunk.content):
                    result.append(
                        MarkdownParsedChunk(
                            content=sub_text,
                            parent_headings=chunk.parent_headings,
                        )
                    )
        return result

    def _split_text_by_tokens(self, text: str) -> list[str]:
        """Split a long text into token-bounded paragraphs."""
        parts = re.split(r"(\n```\n.*?\n```\n)", text, flags=re.DOTALL)
        paragraphs: list[str] = []
        for part in parts:
            if part.startswith("\n```\n") and part.endswith("\n```\n"):
                paragraphs.append(part)
            else:
                paragraphs.extend(re.split(r"\n\s*\n", part))
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        if not paragraphs:
            return [text]

        total_tokens = sum(len(self.tokenizer.encode(p)) for p in paragraphs)
        target = max(1, total_tokens // (total_tokens // self.chunk_max_tokens + 1))

        result: list[str] = []
        current = ""
        current_tokens = 0

        for para in paragraphs:
            para_tokens = len(self.tokenizer.encode(para))
            if current and current_tokens + para_tokens > target:
                result.append(current.strip())
                # Carry the last chunk_overlap_tokens tokens into the next chunk
                encoded = self.tokenizer.encode(current)
                overlap_encoded = encoded[-self.chunk_overlap_tokens :]
                overlap_text = self.tokenizer.decode(overlap_encoded)
                current = overlap_text + "\n\n" + para
                current_tokens = len(self.tokenizer.encode(current))
            else:
                current = (current + "\n\n" + para) if current else para
                current_tokens += para_tokens

        if current:
            result.append(current.strip())

        return result
