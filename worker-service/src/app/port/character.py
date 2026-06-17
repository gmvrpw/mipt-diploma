from typing import Protocol

from src.domain.model import Character


class GetCharacter(Protocol):
    async def __call__(self, path: str) -> Character:
        ...


class SaveCharacter(Protocol):
    async def __call__(self, id: str, character: Character) -> str:
        ...
