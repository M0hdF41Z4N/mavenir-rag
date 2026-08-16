from pathlib import Path

from fastapi import UploadFile

from api.config import settings
from api.exceptions import ValidationError


def validate_uploads(files: list[UploadFile]) -> None:
    """Reject uploads that exceed the file count limit or use unsupported extensions.

    Called at the router layer before any bytes are read from the upload, so bad
    requests fail fast without touching MinIO or consuming memory.
    """
    if len(files) > settings.chat.max_files_per_upload:
        raise ValidationError(f"Too many files. Maximum allowed is {settings.chat.max_files_per_upload}.")
    allowed = set(settings.chat.allowed_extensions)
    for file in files:
        ext = Path(file.filename).suffix.lower()
        if ext not in allowed:
            raise ValidationError(f"File '{file.filename}' has unsupported extension '{ext}'. Allowed: {sorted(allowed)}")
