from __future__ import annotations

from typing import Optional

import httpx
import structlog
from structlog.stdlib import BoundLogger

from .config import RestCancellationConfig

log: BoundLogger = structlog.get_logger(__name__)


class RestCancellationProvider:
    """Pull-style cancellation provider.

    Probes a REST endpoint for each task and returns True iff the resource
    exists (HTTP 200). HTTP 404 means "no cancellation flag set". Other
    response codes are treated as "not cancelled" (fail-open).
    """

    def __init__(self, config: RestCancellationConfig) -> None:
        self._config = config
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self) -> "RestCancellationProvider":
        headers: dict[str, str] = {}
        cert: tuple[str, str] | None = None
        if self._config.authorization is not None:
            auth = self._config.authorization
            if auth.header is not None:
                name, _, value = auth.header.partition(":")
                if value:
                    headers[name.strip()] = value.strip()
                else:
                    headers["Authorization"] = auth.header.strip()
            if auth.certificate is not None:
                cert = (auth.certificate.cert, auth.certificate.key)

        if self._config.method == "POST":
            headers.setdefault("Content-Type", "application/json")

        self._client = httpx.AsyncClient(
            headers=headers,
            cert=cert,
            verify=self._config.verify_ssl,
            timeout=self._config.timeout_seconds,
        )
        log.info("REST cancellation provider started", path=self._config.path,
                 method=self._config.method)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
        log.info("REST cancellation provider stopped", path=self._config.path)

    async def get_task_cancelled(self, task_id: str) -> bool:
        if self._client is None:
            log.warning(
                "REST cancellation provider invoked outside async context; "
                "treating as not cancelled", task_id=task_id,
            )
            return False

        url = self._config.path.format(task_id=task_id)
        try:
            if self._config.method == "GET":
                response = await self._client.get(url)
            else:
                response = await self._client.post(url, json={"id": task_id})
        except httpx.HTTPError as e:
            log.warning(
                "REST cancellation probe failed; treating as not cancelled",
                task_id=task_id, url=url, error=str(e),
            )
            return False

        if response.status_code == 200:
            return True
        if response.status_code == 404:
            return False

        log.warning(
            "REST cancellation probe returned unexpected status; "
            "treating as not cancelled",
            task_id=task_id, url=url, status=response.status_code,
        )
        return False
