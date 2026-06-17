import asyncio
import threading
from typing import Coroutine

import structlog
from structlog.stdlib import BoundLogger


log: BoundLogger = structlog.get_logger(__name__)


class IO:
    def __init__(self) -> None:
        self._controllers = []
        self._started = threading.Event()
        self._stopped = threading.Event()

    def start(self, controllers: list[Coroutine]):
        if self._started.is_set():
            log.warning("IO is running already")
            return

        self._stopped.clear()
        self._started.set()

        self._controllers = controllers

        log.info("IO starting...")

        loop = asyncio.new_event_loop()
        loop.run_until_complete(self._run_async())
        loop.close()

        log.info("IO stopped")

    async def _run_async(self):
        tasks = [asyncio.create_task(c) for c in self._controllers]
        shutdown = asyncio.create_task(asyncio.to_thread(self._stopped.wait))

        _, pending = await asyncio.wait(
            [*tasks, shutdown],
            return_when=asyncio.FIRST_COMPLETED
        )

        for task in pending:
            task.cancel()

        await asyncio.wait(pending, return_when=asyncio.ALL_COMPLETED)

    def stop(self):
        if self._stopped.is_set():
            log.warning("IO is not running yet")
            return

        log.info("Stopping IO...")
        self._stopped.set()
        self._started.clear()

