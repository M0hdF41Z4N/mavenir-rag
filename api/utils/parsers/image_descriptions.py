# Utilities for replacing markdown image placeholders with LLM-generated text descriptions.
# The hybrid parser (hybrid_parser.py) uses the in-memory docling picture items instead,
# calling _is_meaningful_image and _is_valid_description directly from this module.
import logging
import os
import re
import shutil
from io import BytesIO

from PIL import Image as PILImage

from api.client.llm_client import LLMClient
from api.client.minio_client import MinioClient
from api.models import MarkdownParsedPage, TopicEnum
from api.utils.llm_prompts import IMAGE_DESCRIPTION_PROMPT

logger = logging.getLogger(__name__)

_MIN_IMAGE_DIM = 150  # pixels — images smaller than this are likely icons/decorations


def _is_meaningful_image(image_bytes: bytes) -> bool:
    try:
        img = PILImage.open(BytesIO(image_bytes))
        w, h = img.size
        return w >= _MIN_IMAGE_DIM and h >= _MIN_IMAGE_DIM
    except Exception:
        return False


def _is_valid_description(description: str) -> bool:
    return description.strip().upper() != "SKIP"


def replace_images_with_descriptions(
    doc_id: str,
    topic: TopicEnum,
    markdown_pdf_pages: list[MarkdownParsedPage],
    parsed_images_directory: str,
    minio_client: MinioClient,
    llm_client: LLMClient,
) -> list[MarkdownParsedPage]:
    """Replace markdown image placeholders with LLM-generated descriptions.
    Uses a cache to avoid duplicate LLM calls for the same image path.
    """
    try:
        cache: dict[str, str | None] = {}
        for index_num, markdown_pdf_page in enumerate(markdown_pdf_pages):
            page_num = markdown_pdf_page.metadata["page"]
            raw_text = markdown_pdf_page.text
            image_paths = re.findall(r"!\[\]\((.*?)\)", raw_text)

            for img_path in image_paths:
                if img_path in cache:
                    if cache[img_path] is None:
                        raw_text = raw_text.replace(f"![]({img_path})", "")
                        continue
                    description = cache[img_path]
                else:
                    filename = os.path.basename(img_path)
                    object_name = f"parsed_images/{topic.value}/{doc_id}/{filename}"
                    minio_client.upload_file(
                        file_path=img_path,
                        object_name=object_name,
                        metadata={"page_num": page_num},
                    )

                    try:
                        with open(img_path, "rb") as f:
                            img_bytes = f.read()
                        if not _is_meaningful_image(img_bytes):
                            logger.info("Skipping small/decorative image: %s", img_path)
                            cache[img_path] = None
                            raw_text = raw_text.replace(f"![]({img_path})", "")
                            continue
                        # Pass raw bytes — LLMs cannot fetch URLs
                        description = llm_client.chat_with_image(IMAGE_DESCRIPTION_PROMPT, img_bytes)
                    except Exception as e:
                        description = f"Image description failed: {e}"

                    cache[img_path] = description if _is_valid_description(description) else None

                if cache[img_path] is None:
                    raw_text = raw_text.replace(f"![]({img_path})", "")
                    continue

                description = cache[img_path]
                raw_text = raw_text.replace(
                    f"![]({img_path})",
                    f"\n#### **Image Description**\n\n{description}\n",
                )

            markdown_pdf_pages[index_num].text = raw_text

        return markdown_pdf_pages

    except Exception as e:
        logger.error("Failed to add image descriptions: %s", e, exc_info=True)
        raise

    finally:
        if os.path.exists(parsed_images_directory):
            shutil.rmtree(parsed_images_directory)
