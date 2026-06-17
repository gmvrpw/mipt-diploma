from __future__ import annotations

import posixpath
import time

from src import metrics
from src.domain.model import Character, Frames
from src.infra.s3 import AwsStorage, AwsStorageConfig

from .mapping import (
    from_character,
    from_frames,
    to_character,
    to_domain_error,
    to_frames,
)


class S3StorageRepository(AwsStorage):
    def __init__(self, prefix: str, config: AwsStorageConfig) -> None:
        super().__init__(config)
        self._prefix = prefix.rstrip("/")

    async def get(self, key: str) -> bytes:
        started = time.monotonic()
        try:
            data = await super().get(key)
        finally:
            metrics.storage_operation_duration_seconds.labels(
                operation="get").observe(time.monotonic() - started)
        metrics.storage_bytes_in_total.inc(len(data))
        return data

    async def set(self, key: str, data: bytes) -> str:
        started = time.monotonic()
        try:
            result = await super().set(key, data)
        finally:
            metrics.storage_operation_duration_seconds.labels(
                operation="set").observe(time.monotonic() - started)
        metrics.storage_bytes_out_total.inc(len(data))
        return result

    def _key(self, id: str) -> str:
        if not self._prefix:
            return id
        return posixpath.join(self._prefix, id)

    @staticmethod
    def _character_key(id: str) -> str:
        return f"{id}.png"

    @staticmethod
    def _atlas_key(id: str) -> str:
        return f"{id}.png"

    @staticmethod
    def _meta_key(atlas_key: str) -> str:
        if atlas_key.endswith(".png"):
            return f"{atlas_key[:-4]}.json"
        return f"{atlas_key}.json"

    async def get_character(self, path: str) -> Character:
        try:
            return to_character(await self.get(path))
        except BaseException as e:
            raise to_domain_error(e)

    async def save_character(self, id: str, character: Character) -> str:
        try:
            key = self._key(self._character_key(id))
            await self.set(key, from_character(character))
            return key
        except BaseException as e:
            raise to_domain_error(e)

    async def get_frames(self, path: str) -> Frames:
        try:
            atlas_bytes = await self.get(path)
            try:
                meta_bytes = await self.get(self._meta_key(path))
            except BaseException as inner:
                if to_domain_error(inner).__class__.__name__ == "EntityNotFoundError":
                    meta_bytes = None
                else:
                    raise
            return to_frames(atlas_bytes, meta_bytes)
        except BaseException as e:
            raise to_domain_error(e)

    async def save_frames(self, id: str, frames: Frames) -> str:
        try:
            atlas_bytes, meta_bytes = from_frames(frames)
            atlas_key = self._key(self._atlas_key(id))
            await self.set(atlas_key, atlas_bytes)
            await self.set(self._meta_key(atlas_key), meta_bytes)
            return atlas_key
        except BaseException as e:
            raise to_domain_error(e)
