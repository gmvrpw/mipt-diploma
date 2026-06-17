from dataclasses import dataclass

from src.app.port.character import GetCharacter, SaveCharacter
from src.domain.pipeline import StandPipeline
from src.domain.worker import Ticket


@dataclass
class StandRequest:
    task_id: str
    character_path: str
    prompt: str
    pose_path: str | None = None
    negative_prompt: str | None = None


@dataclass
class StandResponse:
    task_id: str
    character_path: str


class StandService:
    def __init__(
        self,
        get_character: GetCharacter,
        save_character: SaveCharacter,
        pipeline: StandPipeline,
    ):
        self._get_character = get_character
        self._save_character = save_character
        self._pipeline = pipeline

    async def __call__(self, request: StandRequest, ticket: Ticket) -> StandResponse:
        character = await self._get_character(request.character_path)
        pose = await self._get_character(request.pose_path) if request.pose_path else None

        async with self._pipeline.pipe(
            character, pose, request.prompt, request.negative_prompt, ticket,
        ) as pipe:
            output = await pipe.pipe()
            character_path = await self._save_character(
                f"{request.task_id}_1", output.character)

            return StandResponse(task_id=request.task_id, character_path=character_path)
