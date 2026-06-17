from typing import Protocol

from src.domain.worker import Ticket


class GetTicket(Protocol):
    async def __call__(self) -> Ticket:
        ...
