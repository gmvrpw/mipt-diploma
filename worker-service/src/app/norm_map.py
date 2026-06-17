from dataclasses import dataclass

from src.app.port.frames import GetFrames, SaveFrames
from src.domain.pipeline import NormMapPipeline
from src.domain.worker import Ticket


@dataclass
class NormMapRequest:
    task_id: str
    frames_path: str


@dataclass
class NormMapResponse:
    task_id: str
    frames_path: str


class NormMapService:
    def __init__(
        self,
        get_frames: GetFrames,
        save_frames: SaveFrames,
        pipeline: NormMapPipeline,
    ):
        self._get_frames = get_frames
        self._save_frames = save_frames
        self._pipeline = pipeline

    async def __call__(self, request: NormMapRequest, ticket: Ticket) -> NormMapResponse:
        frames = await self._get_frames(request.frames_path)

        async with self._pipeline.pipe(frames, ticket) as pipe:
            output = await pipe.pipe()
            frames_path = await self._save_frames(
                f"{request.task_id}_1", output.frames)

            return NormMapResponse(task_id=request.task_id, frames_path=frames_path)
