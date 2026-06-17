import asyncio

import pytest

from src.app.cancellation import (
    CancellationService,
    CancelTaskRequest,
    was_task_cancelled,
)


async def _yes(_id: str) -> bool:
    return True


async def _no(_id: str) -> bool:
    return False


async def _boom(_id: str) -> bool:
    raise RuntimeError("provider broken")


async def test_empty_pull_list_returns_false():
    svc = CancellationService()
    assert await svc._cancelled("t") is False


async def test_pull_fanout_any_true():
    svc = CancellationService(get_task_cancelled=[_no, _yes, _no])
    assert await svc._cancelled("t") is True


async def test_pull_fanout_all_false():
    svc = CancellationService(get_task_cancelled=[_no, _no])
    assert await svc._cancelled("t") is False


async def test_pull_exception_does_not_crash():
    svc = CancellationService(get_task_cancelled=[_boom, _no])
    assert await svc._cancelled("t") is False


async def test_push_cancels_tracked_task():
    svc = CancellationService()

    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def work():
        started.set()
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    tracked = svc.create_cancellable_task("t1", work())
    await started.wait()

    svc(CancelTaskRequest(task_id="t1"))

    with pytest.raises(asyncio.CancelledError):
        await tracked
    assert was_task_cancelled(tracked)
    assert cancelled.is_set()


async def test_pre_scheduled_cancel_short_circuits():
    svc = CancellationService(get_task_cancelled=[_yes])
    started = asyncio.Event()

    async def work():
        started.set()
        await asyncio.sleep(10)

    tracked = svc.create_cancellable_task("t1", work())

    with pytest.raises(asyncio.CancelledError):
        await tracked
    assert was_task_cancelled(tracked)
