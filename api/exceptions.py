# Domain exceptions — raised by services/utilities and caught by the global handler.
# Keeping these separate from FastAPI's HTTPException means business logic never
# imports from `fastapi`, making it easier to unit-test without spinning up the app.
from fastapi import Request
from fastapi.responses import JSONResponse


class DomainError(Exception):
    """Base class for all domain-level errors raised by services and utilities."""

    status_code: int = 500
    detail: str = "An unexpected error occurred."

    def __init__(self, detail: str | None = None):
        self.detail = detail or self.__class__.detail
        super().__init__(self.detail)


class NotFoundError(DomainError):
    status_code = 404
    detail = "The requested resource was not found."


class ValidationError(DomainError):
    status_code = 400
    detail = "Invalid input."


class IngestionError(DomainError):
    status_code = 422
    detail = "Document ingestion failed."


class LLMUnavailableError(DomainError):
    status_code = 503
    detail = "The LLM service is currently unavailable."


class ServiceUnavailableError(DomainError):
    status_code = 503
    detail = "A required service is temporarily unavailable."


async def domain_error_handler(request: Request, exc: DomainError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": type(exc).__name__, "detail": exc.detail},
    )
