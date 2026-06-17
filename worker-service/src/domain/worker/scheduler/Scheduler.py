import asyncio
from queue import Queue
from typing import Optional

from src import metrics

from .Ticket import Ticket


class Scheduler:
    def __init__(self):
        self._queue = Queue[Ticket]()

    async def queue(self) -> Ticket:
        ticket = Ticket()

        await asyncio.to_thread(self._queue.put, ticket)
        metrics.scheduler_queue_size.set(self._queue.qsize())
        return ticket

    def get(self, timeout: Optional[float]) -> Ticket:
        ticket = self._queue.get(timeout=timeout)
        metrics.scheduler_queue_size.set(self._queue.qsize())
        return ticket
