import time


class Timer:
    """Pure stopwatch. start() → stop() → elapsed."""

    def __init__(self) -> None:
        self._start: float = 0.0
        self._frozen: float | None = None

    def start(self) -> None:
        self._start = time.monotonic()
        self._frozen = None

    def stop(self) -> float:
        self._frozen = time.monotonic() - self._start
        return self._frozen

    @property
    def elapsed(self) -> float:
        if self._frozen is not None:
            return self._frozen
        return time.monotonic() - self._start
