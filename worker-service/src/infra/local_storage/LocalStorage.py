from typing import TypedDict
from typing_extensions import Unpack
import aiofiles
import asyncio
import os


class LocalStorageConfig(TypedDict):
    read_timeout: float
    read_retries: int
    write_timeout: float
    write_retries: int


class LocalStorage:
    def __init__(self, **config: Unpack[LocalStorageConfig]):
        self._read_timeout = config['read_timeout']
        self._read_retries = config['read_retries']

        self._write_timeout = config['write_timeout']
        self._write_retries = config['write_retries']

    async def get(self, path: str) -> bytes:
        exc: BaseException | None = None

        for _ in range(self._read_retries + 1):
            try:
                return await asyncio.wait_for(self._read(path), timeout=self._read_timeout)
            except BaseException as e:
                exc = e

        assert exc is not None
        raise exc

    async def set(self, path: str, data: bytes) -> str:
        exc: BaseException | None = None

        for _ in range(self._write_retries + 1):
            try:
                return await asyncio.wait_for(self._write(path, data), timeout=self._write_timeout)
            except BaseException as e:
                exc = e

        assert exc is not None
        raise exc

    @staticmethod
    async def _read(full_path: str) -> bytes:
        async with aiofiles.open(full_path, "rb") as f:
            return await f.read()

    @staticmethod
    async def _write(path: str, data: bytes) -> str:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        async with aiofiles.open(path, "wb") as f:
            await f.write(data)
        return path
