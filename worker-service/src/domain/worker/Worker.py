from queue import Empty
import threading

import structlog
from structlog.stdlib import BoundLogger


from src.domain.worker.device.Device import Device
from src.domain.worker.scheduler import Scheduler, TicketCanceledError

log: BoundLogger = structlog.get_logger(__name__)


class Worker:
    def __init__(self, scheduler: Scheduler, device: Device) -> None:
        self._scheduler = scheduler
        self._device = device

        self._started = threading.Event()
        self._stopped = threading.Event()

    def start(self):
        if self._started.is_set():
            log.warning("Worker is running already")
            return

        self._stopped.clear()
        self._started.set()

        log.info("Worker starting...")

        while not self._stopped.is_set():
            try:
                self._scheduler \
                    .get(timeout=1) \
                    .task(timeout=1)()
                self._device.cache_all()
            except Empty:
                pass
            except TimeoutError:
                pass
            except TicketCanceledError:
                pass

        log.info("Worker stopped")

    def stop(self):
        if self._stopped.is_set():
            log.warning("Worker is not running yet")
            return

        log.info("Stopping worker...")
        self._stopped.set()
        self._started.clear()

