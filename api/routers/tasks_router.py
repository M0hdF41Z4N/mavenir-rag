# Background task status endpoints. Task metadata (status, progress, result, error)
# is stored as a Redis hash under the key "task:<task_id>" with a configurable TTL.
# These endpoints let clients poll for the result of async ingestion tasks.
import logging

from fastapi import APIRouter, Depends, Query

from api.dependencies import get_task_manager
from api.exceptions import ServiceUnavailableError
from api.models import (
    RAGIngestionResponse,
    TaskListResponse,
    TaskStatusEnum,
    TaskStatusResponse,
)
from api.services.task_manager import TaskManager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/corpus", tags=["Tasks"])


@router.get("/tasks", response_model=TaskListResponse)
def list_tasks(
    task_ids: list[str] = Query(  # noqa: B008
        default=[], description="Task IDs to fetch; omit to return all"
    ),
    task_manager: TaskManager = Depends(get_task_manager),  # noqa: B008
) -> TaskListResponse:
    """Get status of background ingestion tasks.

    Omit `task_ids` to list all tasks. Pass one or more `task_ids` to fetch specific tasks.
    """
    if not task_ids:
        try:
            raw_tasks = task_manager.list_tasks()
        except Exception as e:
            logger.warning("Failed to fetch tasks from Redis: %s", e)
            return TaskListResponse(tasks=[], total=0)

        tasks = [
            TaskStatusResponse(
                task_id=t["task_id"],
                status=TaskStatusEnum(t["status"]),
                description=t.get("description") or None,
                progress=t.get("progress") or None,
                elapsed_time=t.get("elapsed_time"),
                error=t.get("error") or None,
                created_at=t.get("created_at"),
                updated_at=t.get("updated_at"),
            )
            for t in raw_tasks
        ]
        return TaskListResponse(tasks=tasks, total=len(tasks))

    results: list[TaskStatusResponse] = []
    not_found: list[str] = []

    for tid in task_ids:
        try:
            task = task_manager.get_task(tid)
        except Exception as e:
            logger.warning("Failed to fetch task %s from Redis: %s", tid, e)
            raise ServiceUnavailableError("Task service temporarily unavailable") from e

        if task is None:
            not_found.append(tid)
            continue

        result = None
        if task.get("result"):
            result = RAGIngestionResponse.model_validate_json(task["result"])

        results.append(
            TaskStatusResponse(
                task_id=tid,
                status=TaskStatusEnum(task["status"]),
                description=task.get("description") or None,
                progress=task.get("progress") or None,
                elapsed_time=task.get("elapsed_time"),
                result=result,
                error=task.get("error") or None,
                created_at=task.get("created_at"),
                updated_at=task.get("updated_at"),
            )
        )

    return TaskListResponse(tasks=results, total=len(results), not_found=not_found)
