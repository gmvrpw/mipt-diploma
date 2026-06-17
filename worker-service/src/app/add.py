from dataclasses import dataclass

from src.app.port.character import GetCharacter, SaveCharacter
from src.domain.pipeline import AddPipeline
from src.domain.worker import Ticket


@dataclass
class AddRequest:
    task_id: str
    character_path: str
    asset_path: str
    prompt: str
    negative_prompt: str | None = None


@dataclass
class AddResponse:
    task_id: str
    character_path: str


class AddService:
    def __init__(
        self,
        get_character: GetCharacter,
        save_character: SaveCharacter,
        pipeline: AddPipeline,
    ):
        self._get_character = get_character
        self._save_character = save_character
        self._pipeline = pipeline

    async def __call__(self, request: AddRequest, ticket: Ticket) -> AddResponse:
        character = await self._get_character(request.character_path)
        asset = await self._get_character(request.asset_path)

        async with self._pipeline.pipe(
            character, asset, request.prompt, request.negative_prompt, ticket,
        ) as pipe:
            output = await pipe.pipe()
            character_path = await self._save_character(
                f"{request.task_id}_1", output.character)

            return AddResponse(task_id=request.task_id, character_path=character_path)
