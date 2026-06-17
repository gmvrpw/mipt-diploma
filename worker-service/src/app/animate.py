from dataclasses import dataclass

from src.app.port.character import GetCharacter
from src.app.port.frames import SaveFrames
from src.domain.pipeline import AnimatePipeline
from src.domain.worker import Ticket


@dataclass
class AnimateRequest:
    task_id: str
    first_frame_path: str
    prompt: str
    last_frame_path: str | None = None
    num_frames: int = 45
    loop: bool | None = None


@dataclass
class AnimateResponse:
    task_id: str
    frames_path: str


class AnimateService:
    def __init__(
        self,
        get_character: GetCharacter,
        save_frames: SaveFrames,
        pipeline: AnimatePipeline,
    ):
        self._get_character = get_character
        self._save_frames = save_frames
        self._pipeline = pipeline

    async def __call__(self, request: AnimateRequest, ticket: Ticket) -> AnimateResponse:
        first_frame = await self._get_character(request.first_frame_path)
        last_frame = (
            await self._get_character(request.last_frame_path)
            if request.last_frame_path else None
        )

        async with self._pipeline.pipe(
            first_frame, last_frame, request.prompt,
            request.num_frames, request.loop, ticket,
        ) as pipe:
            output = await pipe.pipe()
            frames_path = await self._save_frames(
                f"{request.task_id}_1", output.frames)

            return AnimateResponse(task_id=request.task_id, frames_path=frames_path)
