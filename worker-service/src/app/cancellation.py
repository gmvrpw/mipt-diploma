import asyncio
from asyncio import Task, get_event_loop
from dataclasses import dataclass
from typing import Coroutine

from src.app.port.task import GetTaskCancelled

TASK_CANCELLED = object()  # Legacy sentinel; retained for backward-compat
                           # callers that still check exception args.

_CANCEL_FLAG_ATTR = "_worker_cancellation_triggered"


@dataclass
class CancelTaskRequest:
    task_id: str


def was_task_cancelled(task: Task) -> bool:
    """Returns True if the task was cancelled via CancellationService.

    Distinguishes our cancellation from an externally-triggered one (e.g.
    the asyncio loop shutting down). asyncio strips ``CancelledError`` args
    when propagating through a task, so callers cannot rely on the sentinel
    being present on the exception itself.
    """
    return getattr(task, _CANCEL_FLAG_ATTR, False) is True


class CancellationService:
    """Tracks running tasks for push-cancellation and asks pull-providers
    whether a task is already cancelled before scheduling.
    """

    def __init__(
        self,
        get_task_cancelled: list[GetTaskCancelled] | None = None,
    ):
        self._get_task_cancelled = list(get_task_cancelled or [])
        # Inner workload tasks, keyed by task_id.
        self._inner_tasks: dict[str, Task] = {}
        # Outer wrapper tasks (the cancellable tasks returned to the caller),
        # so we can tag them on cancellation.
        self._outer_tasks: dict[str, Task] = {}

    def __call__(self, request: CancelTaskRequest):
        task_id = request.task_id
        outer = self._outer_tasks.get(task_id)
        if outer is not None:
            setattr(outer, _CANCEL_FLAG_ATTR, True)
        inner = self._inner_tasks.get(task_id)
        if inner is not None:
            inner.cancel(TASK_CANCELLED)

    def create_cancellable_task(self, id: str, coro: Coroutine):
        outer = get_event_loop().create_task(self._track(id, coro))
        self._outer_tasks[id] = outer
        return outer

    async def _track(self, id: str, coro: Coroutine):
        task_is_cancelled = await self._cancelled(id)

        inner = get_event_loop().create_task(coro)
        self._inner_tasks[id] = inner

        outer = asyncio.current_task()
        if task_is_cancelled and outer is not None:
            setattr(outer, _CANCEL_FLAG_ATTR, True)
            inner.cancel(TASK_CANCELLED)

        try:
            await inner
        finally:
            self._inner_tasks.pop(id, None)
            self._outer_tasks.pop(id, None)

    async def _cancelled(self, id: str) -> bool:
        if not self._get_task_cancelled:
            return False
        results = await asyncio.gather(
            *(provider(id) for provider in self._get_task_cancelled),
            return_exceptions=True,
        )
        return any(r is True for r in results)
