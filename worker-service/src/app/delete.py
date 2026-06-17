from dataclasses import dataclass

from src.app.port.character import GetCharacter, SaveCharacter
from src.domain.pipeline import DeletePipeline
from src.domain.worker import Ticket


@dataclass
class DeleteRequest:
    task_id: str
    character_path: str
    prompt: str
    negative_prompt: str | None = None


@dataclass
class DeleteResponse:
    task_id: str
    character_path: str


class DeleteService:
    def __init__(
        self,
        get_character: GetCharacter,
        save_character: SaveCharacter,
        pipeline: DeletePipeline,
    ):
        self._get_character = get_character
        self._save_character = save_character
        self._pipeline = pipeline

    async def __call__(self, request: DeleteRequest, ticket: Ticket) -> DeleteResponse:
        character = await self._get_character(request.character_path)

        async with self._pipeline.pipe(
            character, request.prompt, request.negative_prompt, ticket,
        ) as pipe:
            output = await pipe.pipe()
            character_path = await self._save_character(
                f"{request.task_id}_1", output.character)

            return DeleteResponse(task_id=request.task_id, character_path=character_path)
