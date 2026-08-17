import logging
import traceback
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse

from api.client.llm_client import LLMClient
from api.client.milvus_client import MilvusClient
from api.client.minio_client import MinioClient
from api.client.redis_client import RedisClient
from api.config import settings
from api.exceptions import DomainError, domain_error_handler
from api.routers import corpus_router, ingestion_router, matcher_router, tasks_router
from api.services import get_hello_message
from api.services.corpus_storage import CorpusStorageService
from api.services.ingestion_service import IngestionService
from api.services.reranker import CrossEncoderReranker
from api.services.task_manager import TaskManager
from api.utils.markdown_parser import MarkdownParser

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create single instances of LLMLite, Milvus and MinIO
    # Access them in router files using request.app.state...
    app.state.minio_client = MinioClient(settings=settings.minio)
    app.state.llm_client = LLMClient(settings=settings.llm)
    app.state.milvus_client = MilvusClient(settings=settings.milvus)
    app.state.redis_client = RedisClient(settings=settings.redis)
    app.state.markdown_parser = MarkdownParser(parser_settings=settings.doc_parser)
    app.state.settings = settings
    app.state.corpus_storage = CorpusStorageService(minio_client=app.state.minio_client)
    # Cheap to construct — the cross-encoder model is lazy-loaded on first rerank, so
    # this touches no heavy deps while reranking is disabled.
    app.state.reranker = CrossEncoderReranker(settings=settings.rerank)
    app.state.task_manager = TaskManager(
        redis=app.state.redis_client.redis,
        ttl_seconds=settings.background.task_result_ttl_seconds,
    )
    app.state.ingestion_service = IngestionService(
        minio_client=app.state.minio_client,
        milvus_client=app.state.milvus_client,
        llm_client=app.state.llm_client,
        markdown_parser=app.state.markdown_parser,
        task_manager=app.state.task_manager,
        corpus_storage=app.state.corpus_storage,
        settings=settings,
    )
    try:
        app.state.task_manager.mark_stale_tasks_failed()
    except Exception as e:
        logger.warning("Could not clean up stale tasks on startup: %s", e)
    yield


app = FastAPI(title="Mavenir Rag FastAPI App", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.security.cors_allowed_origins,
    # allow_credentials must be False when origins contains "*"
    allow_credentials=settings.security.cors_allowed_origins != ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception: %s", exc)
    logger.error(traceback.format_exc())
    content = {"detail": "Internal Server Error"}
    if settings.security.expose_exception_details:
        content["message"] = str(exc)
    return JSONResponse(status_code=500, content=content)


app.add_exception_handler(DomainError, domain_error_handler)

app.include_router(corpus_router.router)
app.include_router(ingestion_router.router)
app.include_router(tasks_router.router)
app.include_router(matcher_router.router)


def _custom_openapi():
    """Generate and cache the OpenAPI schema with a Bearer auth scheme.

    FastAPI's default schema does not declare a security scheme for our
    Keycloak-protected endpoints, so Swagger UI shows no **Authorize** button
    and callers cannot attach a token from the docs page. This override
    augments the generated schema with a `BearerAuth` (HTTP bearer, JWT)
    entry under `components.securitySchemes`, enabling the **Authorize** flow
    in Swagger UI. The result is cached on `app.openapi_schema` so the schema
    is built only once per process. Installed via `app.openapi = _custom_openapi`.
    """
    if app.openapi_schema:
        return app.openapi_schema
    schema = get_openapi(
        title=app.title,
        version=app.version,
        description=(
            "Mavenir Rag document matching and RAG API.\n\n"
            "**Authentication:** Protected endpoints require a Keycloak Bearer token. "
            "Call `POST /mavenir-rag/auth/token` with a Keycloak refresh token to obtain one, "
            "then click **Authorize** and paste the returned `access_token` (without the 'Bearer' prefix)."
        ),
        routes=app.routes,
    )
    bearer = schema.setdefault("components", {}).setdefault("securitySchemes", {}).setdefault("BearerAuth", {})
    bearer.update(
        {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
            "description": "Keycloak access token. Obtain one via POST /mavenir-rag/auth/token.",
        }
    )
    app.openapi_schema = schema
    return app.openapi_schema


app.openapi = _custom_openapi


@app.get("/")
def read_root():
    return get_hello_message()


def start():
    uvicorn.run("api:app", host="0.0.0.0", port=settings.app_port, reload=True)  # noqa: S104


# Ref: https://fastapi.tiangolo.com/tutorial/debugging/?h=uvi#call-uvicorn
def serve():
    uvicorn.run("api:app", host="0.0.0.0", port=settings.app_port)  # noqa: S104
