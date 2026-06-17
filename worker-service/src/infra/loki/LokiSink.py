from __future__ import annotations

import asyncio
import json
import queue
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Optional

import httpx
import structlog
from structlog.stdlib import BoundLogger

log: BoundLogger = structlog.get_logger(__name__)


@dataclass
class LokiSinkConfig:
    url: str
    endpoint: str = "/loki/api/v1/push"
    tenant_id: Optional[str] = None
    auth: Optional[tuple[str, str]] = None
    labels: dict[str, str] = field(default_factory=dict)
    timeout_seconds: float = 5.0
    batch_size: int = 100
    batch_interval_seconds: float = 1.0
    retries: int = 3
    retry_backoff_seconds: float = 0.5
    verify_ssl: bool = True


class LokiSink:
    """Thread-safe ingest, async batched push.

    Producer side (``enqueue``) is callable from any thread (typical caller:
    a structlog processor on the main thread). Consumer side is an asyncio
    flusher started in ``__aenter__`` and torn down in ``__aexit__``.
    """

    def __init__(self, config: LokiSinkConfig) -> None:
        self._config = config
        self._queue: queue.Queue[tuple[int, str]] = queue.Queue(maxsize=10_000)
        self._stop = threading.Event()
        self._client: Optional[httpx.AsyncClient] = None
        self._flusher_task: Optional[asyncio.Task] = None

    @property
    def url(self) -> str:
        return self._config.url.rstrip("/") + self._config.endpoint

    def enqueue(self, event_dict: dict[str, Any]) -> None:
        """Push a structured event onto the queue. Drops on overflow."""
        timestamp_ns = _event_timestamp_ns(event_dict)
        try:
            line = json.dumps(event_dict, default=str, ensure_ascii=False)
        except Exception:
            return
        try:
            self._queue.put_nowait((timestamp_ns, line))
        except queue.Full:
            return

    async def __aenter__(self) -> "LokiSink":
        headers = {"Content-Type": "application/json"}
        if self._config.tenant_id:
            headers["X-Scope-OrgID"] = self._config.tenant_id

        auth = httpx.BasicAuth(*self._config.auth) if self._config.auth else None
        transport = httpx.AsyncHTTPTransport(retries=max(0, self._config.retries))

        self._client = httpx.AsyncClient(
            base_url=self._config.url.rstrip("/"),
            headers=headers,
            auth=auth,
            verify=self._config.verify_ssl,
            timeout=self._config.timeout_seconds,
            transport=transport,
        )
        self._stop.clear()
        self._flusher_task = asyncio.create_task(self._flush_loop())
        log.info("Loki sink started", url=self.url)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        if self._flusher_task is not None:
            try:
                await asyncio.wait_for(self._flusher_task, timeout=self._config.batch_interval_seconds + 2.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._flusher_task.cancel()
            self._flusher_task = None

        await self._drain_and_push_remaining()

        if self._client is not None:
            await self._client.aclose()
            self._client = None
        log.info("Loki sink stopped", url=self.url)

    async def _flush_loop(self) -> None:
        while not self._stop.is_set():
            batch = await asyncio.to_thread(self._collect_batch)
            if batch:
                await self._push(batch)

    def _collect_batch(self) -> list[tuple[int, str]]:
        try:
            first = self._queue.get(timeout=self._config.batch_interval_seconds)
        except queue.Empty:
            return []
        batch = [first]
        deadline = time.monotonic() + self._config.batch_interval_seconds
        while len(batch) < self._config.batch_size:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                batch.append(self._queue.get(timeout=remaining))
            except queue.Empty:
                break
        return batch

    async def _drain_and_push_remaining(self) -> None:
        batch: list[tuple[int, str]] = []
        while True:
            try:
                batch.append(self._queue.get_nowait())
            except queue.Empty:
                break
            if len(batch) >= self._config.batch_size:
                await self._push(batch)
                batch = []
        if batch:
            await self._push(batch)

    async def _push(self, batch: list[tuple[int, str]]) -> None:
        if self._client is None:
            return
        payload = {
            "streams": [{
                "stream": self._config.labels,
                "values": [[str(ts), line] for ts, line in batch],
            }],
        }
        body = json.dumps(payload, ensure_ascii=False)
        delay = self._config.retry_backoff_seconds
        for attempt in range(1, max(1, self._config.retries) + 1):
            try:
                response = await self._client.post(
                    self._config.endpoint, content=body,
                )
                if 200 <= response.status_code < 300:
                    return
                log.warning(
                    "Loki push failed",
                    url=self.url, status=response.status_code,
                    attempt=attempt,
                )
            except httpx.HTTPError as e:
                log.warning(
                    "Loki push raised",
                    url=self.url, attempt=attempt, error=str(e),
                )
            if attempt < self._config.retries:
                await asyncio.sleep(delay)
                delay *= 2


def _event_timestamp_ns(event_dict: dict[str, Any]) -> int:
    ts = event_dict.get("timestamp")
    if isinstance(ts, (int, float)):
        return int(float(ts) * 1_000_000_000)
    return time.time_ns()


@asynccontextmanager
async def open_sinks(sinks: list[LokiSink]) -> AsyncIterator[list[LokiSink]]:
    """Open all sinks concurrently inside an async-context."""
    entered: list[LokiSink] = []
    try:
        for sink in sinks:
            await sink.__aenter__()
            entered.append(sink)
        yield entered
    finally:
        for sink in reversed(entered):
            await sink.__aexit__(None, None, None)
