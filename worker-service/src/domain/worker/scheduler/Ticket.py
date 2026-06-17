import threading
from typing import Callable, Optional


class TicketCanceledError(Exception):
    """Raised on operating with closed ticket."""
    pass


class TicketAlreadyUsedError(Exception):
    """Raised on ticket reusing."""
    pass


class Ticket:
    def __init__(self):
        self._used = threading.Condition()
        self._task: Callable | None = None
        self._canceled: bool = False

    def use(self, task: Callable):
        with self._used:
            if self._task is not None:
                raise TicketAlreadyUsedError("Cannot reuse ticket")
            if self._canceled:
                raise TicketCanceledError("Cannot use canceled ticket")
            self._task = task
            self._used.notify_all()

    def task(self, timeout: Optional[float] = None):
        with self._used:
            while self._task is None:
                if self._canceled:
                    raise TicketCanceledError(
                        "Cannot get task from canceled ticket")
                if not self._used.wait(timeout=timeout) and self._task is None:
                    raise TimeoutError("Timed out waiting for ticket task")
            return self._task

    def cancel(self):
        with self._used:
            self._canceled = True
            self._used.notify_all()
