from src.domain.worker import Ticket

from src.app.port.ticket import GetTicket


class TicketService:
    def __init__(self, get_ticket: GetTicket):
        self._get_ticket = get_ticket

    async def __call__(self) -> Ticket:
        return await self._get_ticket()
