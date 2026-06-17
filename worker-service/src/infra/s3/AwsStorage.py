from __future__ import annotations

from contextlib import AsyncExitStack
from typing import Any, TypedDict

import aioboto3
from botocore.config import Config as BotocoreConfig


class AwsStorageConfig(TypedDict, total=False):
    service_name: str
    bucket: str
    region_name: str | None
    endpoint_url: str | None
    api_version: str | None
    use_ssl: bool | None
    verify: bool | None
    # Session-level credentials
    profile_name: str | None
    aws_access_key_id: str | None
    aws_secret_access_key: str | None
    aws_session_token: str | None
    # Botocore client config
    botocore_config: BotocoreConfig | None


class AwsStorage:
    def __init__(self, config: AwsStorageConfig) -> None:
        self._service_name: str = config.get("service_name", "s3")
        self._bucket: str = config["bucket"]

        self._session = aioboto3.Session(
            aws_access_key_id=config.get("aws_access_key_id"),
            aws_secret_access_key=config.get("aws_secret_access_key"),
            aws_session_token=config.get("aws_session_token"),
            region_name=config.get("region_name"),
            profile_name=config.get("profile_name"),
        )

        self._client_kwargs: dict[str, Any] = {}
        for key in ("region_name", "api_version", "use_ssl", "verify", "endpoint_url"):
            value = config.get(key)
            if value is not None:
                self._client_kwargs[key] = value
        botocore_config = config.get("botocore_config")
        if botocore_config is not None:
            self._client_kwargs["config"] = botocore_config

        self._stack: AsyncExitStack | None = None
        self._client: Any = None

    async def __aenter__(self) -> "AwsStorage":
        self._stack = AsyncExitStack()
        await self._stack.__aenter__()
        self._client = await self._stack.enter_async_context(
            self._session.client(self._service_name, **self._client_kwargs)
        )
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._stack is not None:
            await self._stack.__aexit__(exc_type, exc, tb)
            self._stack = None
            self._client = None

    async def get(self, key: str) -> bytes:
        assert self._client is not None, "AwsStorage used outside async context"
        response = await self._client.get_object(Bucket=self._bucket, Key=key)
        async with response["Body"] as stream:
            return await stream.read()

    async def set(self, key: str, data: bytes) -> str:
        assert self._client is not None, "AwsStorage used outside async context"
        await self._client.put_object(Bucket=self._bucket, Key=key, Body=data)
        return key
