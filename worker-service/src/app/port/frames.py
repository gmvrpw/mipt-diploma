from typing import Protocol

from src.domain.model import Frames


class GetFrames(Protocol):
    async def __call__(self, path: str) -> Frames:
        ...


class SaveFrames(Protocol):
    async def __call__(self, id: str, frames: Frames) -> str:
        ...
