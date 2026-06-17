from typing import Protocol


class GetTaskCancelled(Protocol):
    async def __call__(self, task_id: str) -> bool:
        ...
