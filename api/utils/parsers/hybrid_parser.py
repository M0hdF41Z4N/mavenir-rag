# Hybrid document parsing pipeline:
#   1. pymupdf4llm  — fast, accurate text/markdown extraction per page
#   2. docling       — structured table extraction (Markdown) + figure extraction with LLM descriptions
# Two libraries are used because pymupdf4llm is faster for text but lacks table structure,
# while docling provides rich structured output but is slower and RAM-heavy (~500MB–1GB for models).
# pymupdf4llm output is the base; docling tables and image descriptions are appended per-page.
import logging
import tempfile
from io import BytesIO
from pathlib import Path

import pymupdf
import pymupdf4llm

from api.client.llm_client import LLMClient
from api.client.minio_client import MinioClient
from api.models import IngestedDocumentResponse, MarkdownParsedPage, TopicEnum
from api.utils.ingested_document_builder import build_ingested_document
from api.utils.markdown_parser import MarkdownParser
from api.utils.parsers.image_descriptions import (
    IMAGE_DESCRIPTION_PROMPT,
    _is_meaningful_image,
    _is_valid_description,
)

logger = logging.getLogger(__name__)


def _merge_docling_content(
    pages: list[MarkdownParsedPage],
    docling_doc,
    doc_id: str,
    topic: TopicEnum,
    minio_client: MinioClient,
    llm_client: LLMClient,
) -> list[MarkdownParsedPage]:
    """Enrich pymupdf4llm markdown pages with docling tables and image descriptions.
    pymupdf4llm is the structural skeleton; docling content is appended per-page.
    """
    from docling_core.types.doc import PictureItem, TableItem

    tables_by_page: dict[int, list] = {}
    pictures_by_page: dict[int, list] = {}

    # Build page-indexed lookups so we can merge content in a single pass per page
    for element, _level in docling_doc.iterate_items():
        if isinstance(element, TableItem):
            for prov in element.prov:
                tables_by_page.setdefault(prov.page_no, []).append(element)
        elif isinstance(element, PictureItem):
            for prov in element.prov:
                pictures_by_page.setdefault(prov.page_no, []).append(element)

    total_images = sum(len(pics) for pics in pictures_by_page.values())
    logger.info("[DocParser] Total images extracted to be processed: %d", total_images)

    for page in pages:
        page_num = page.metadata["page"]
        enriched_text = page.text

        for tbl_idx, table in enumerate(tables_by_page.get(page_num, [])):
            table_md = table.export_to_markdown(doc=docling_doc)
            enriched_text += f"\n\n#### Table {tbl_idx + 1}\n\n{table_md}\n"

        for fig_idx, picture in enumerate(pictures_by_page.get(page_num, [])):
            # Extract image bytes from docling's picture item
            img_bytes = None
            if picture.image and picture.image.pil_image:
                buf = BytesIO()
                picture.image.pil_image.save(buf, format="PNG")
                img_bytes = buf.getvalue()

            docling_caption = ""
            if hasattr(picture, "caption_text") and picture.caption_text:
                docling_caption = picture.caption_text
            elif hasattr(picture, "captions") and picture.captions:
                docling_caption = " ".join(cap.text for cap in picture.captions if hasattr(cap, "text"))

            # Always prefer LLM for description — docling captions can hallucinate; only skip decorative images
            if img_bytes and _is_meaningful_image(img_bytes):
                prompt = IMAGE_DESCRIPTION_PROMPT
                if docling_caption:
                    prompt += f"\nThe document caption for this image is: '{docling_caption}'"
                try:
                    description = llm_client.chat_with_image(prompt, img_bytes)
                except Exception as e:
                    description = f"Image description failed: {e}"

                if not _is_valid_description(description):
                    logger.info(
                        "Skipping non-meaningful image on page %d, fig %d",
                        page_num,
                        fig_idx + 1,
                    )
                    continue
            elif docling_caption:
                description = docling_caption
            else:
                logger.info(
                    "Skipping small/decorative image on page %d, fig %d",
                    page_num,
                    fig_idx + 1,
                )
                continue

            enriched_text += f"\n\n#### **Image Description**\n\n{description}\n"

        page.text = enriched_text

    return pages


def process_document_for_tasks(
    doc_id: str,
    topic: TopicEnum,
    file_name: str,
    minio_client: MinioClient,
    llm_client: LLMClient,
    markdown_parser: MarkdownParser,
) -> IngestedDocumentResponse:
    """Hybrid pipeline: pymupdf4llm for fast text extraction + docling for structured tables/images.
    Combines speed of pymupdf4llm with accuracy of docling for comprehensive content extraction.
    """
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.document import DocumentStream
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    suffix = Path(file_name).suffix or ".pdf"

    logger.info("[DocParser] Downloading '%s' from MinIO …", file_name)
    pdf_bytes = minio_client.download_bytes(file_name)
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")

    # pymupdf4llm requires a file path — write a temp file (only local I/O, bytes stay in memory for docling)
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    try:
        logger.info("[DocParser] Extracting text layer via pymupdf4llm …")
        markdown_pdf_pages = [MarkdownParsedPage(**page) for page in pymupdf4llm.to_markdown(tmp_path, write_images=False, page_chunks=True)]

        logger.info("[DocParser] Extracting tables & figures via docling …")
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_table_structure = True
        # Skip OCR since pymupdf4llm already extracted text; docling handles structured elements only
        pipeline_options.do_ocr = False
        pipeline_options.generate_picture_images = True

        converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)})
        docling_result = converter.convert(DocumentStream(name=file_name, stream=BytesIO(pdf_bytes)))

        logger.info("[DocParser] Merging docling content into pages …")
        markdown_pdf_pages = _merge_docling_content(
            pages=markdown_pdf_pages,
            docling_doc=docling_result.document,
            doc_id=doc_id,
            topic=topic,
            minio_client=minio_client,
            llm_client=llm_client,
        )

        logger.info("[DocParser] Chunking and embedding …")
        chunks = markdown_parser.chunk_pdf_pages(pages=markdown_pdf_pages)

        try:
            metadata = minio_client.get_metadata(file_name)
            original_filename = metadata.get("x-amz-meta-original-filename", Path(file_name).name)
        except Exception:
            original_filename = Path(file_name).name

        result = build_ingested_document(chunks, llm_client, doc_id, original_filename, log_prefix="[DocParser] ")
        logger.info(
            "[DocParser] Done — %d chunks embedded for doc_id=%s",
            len(result.chunks),
            doc_id,
        )
        return result

    except Exception as e:
        logger.error("[DocParser] Failed to process '%s': %s", file_name, e, exc_info=True)
        raise RuntimeError(f"Failed to process document '{file_name}': {e}") from e

    finally:
        Path(tmp_path).unlink(missing_ok=True)
        doc.close()
