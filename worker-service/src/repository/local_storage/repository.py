from pathlib import Path
from typing_extensions import Unpack

from src.domain.model import Frames, Character

from src.infra.local_storage import LocalStorage, LocalStorageConfig

from .mapping import from_character, to_character, from_frames, to_frames, to_domain_error


class LocalStorageRepository(LocalStorage):
    def __init__(self, base_path: str, **local_storage_config: Unpack[LocalStorageConfig]):
        super().__init__(**local_storage_config)
        self._base_path = base_path

    @staticmethod
    def _character_path(id: str) -> str:
        return f"{id}.png"

    @staticmethod
    def _atlas_path(id: str) -> str:
        return f"{id}.png"

    @staticmethod
    def _meta_path(atlas_path: str) -> str:
        return str(Path(atlas_path).with_suffix(".json"))

    def _resolve(self, path: str) -> str:
        return str(Path(self._base_path) / path)

    async def get_character(self, path: str) -> Character:
        try:
            return to_character(await self.get(path))
        except BaseException as e:
            raise to_domain_error(e)

    async def save_character(self, id: str, character: Character) -> str:
        try:
            path = self._character_path(self._resolve(id))

            await self.set(path, from_character(character))
            return path
        except BaseException as e:
            raise to_domain_error(e)

    async def get_frames(self, path: str) -> Frames:
        try:
            atlas_bytes = await self.get(path)
            try:
                meta_bytes = await self.get(self._meta_path(path))
            except FileNotFoundError:
                meta_bytes = None
            return to_frames(atlas_bytes, meta_bytes)
        except BaseException as e:
            raise to_domain_error(e)

    async def save_frames(self, id: str, frames: Frames) -> str:
        try:
            atlas_bytes, meta_bytes = from_frames(frames)
            atlas_path = self._atlas_path(self._resolve(id))
            await self.set(atlas_path, atlas_bytes)
            await self.set(self._meta_path(atlas_path), meta_bytes)
            return atlas_path
        except BaseException as e:
            raise to_domain_error(e)
