from __future__ import annotations

import threading
import time
from http.server import HTTPServer
from typing import Optional

import structlog
from prometheus_client import start_http_server
from structlog.stdlib import BoundLogger

log: BoundLogger = structlog.get_logger(__name__)

_MAX_ATTEMPTS = 3
_BACKOFF_BASE_SECONDS = 1.0
_SUPERVISOR_POLL_SECONDS = 1.0


class PrometheusServer:
    def __init__(self, host: str = "0.0.0.0", port: int = 9100) -> None:
        self._host = host
        self._port = port

        self._server: Optional[HTTPServer] = None
        self._server_thread: Optional[threading.Thread] = None

        self._supervisor: Optional[threading.Thread] = None
        self._stop = threading.Event()

    @property
    def endpoint(self) -> str:
        return f"http://{self._host}:{self._port}/metrics"

    def start(self) -> None:
        if self._supervisor is not None:
            log.warning("Prometheus server already started", endpoint=self.endpoint)
            return

        self._stop.clear()
        self._bring_up()
        self._supervisor = threading.Thread(
            target=self._supervise, name="prometheus-supervisor", daemon=True,
        )
        self._supervisor.start()

    def stop(self) -> None:
        if self._supervisor is None:
            return

        self._stop.set()
        self._shutdown_server()
        self._supervisor.join(timeout=5.0)
        self._supervisor = None
        log.info("Prometheus server stopped", endpoint=self.endpoint)

    def _bring_up(self) -> None:
        last_error: Optional[BaseException] = None
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                self._server, self._server_thread = start_http_server(
                    port=self._port, addr=self._host,
                )
                log.info("Prometheus server started", endpoint=self.endpoint)
                return
            except BaseException as e:
                last_error = e
                log.warning(
                    "Prometheus server bring-up attempt failed",
                    endpoint=self.endpoint, attempt=attempt, error=str(e),
                )
                time.sleep(_BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)))

        assert last_error is not None
        raise RuntimeError(
            f"Prometheus server failed to start at {self.endpoint} after "
            f"{_MAX_ATTEMPTS} attempts: {last_error}"
        ) from last_error

    def _supervise(self) -> None:
        while not self._stop.is_set():
            time.sleep(_SUPERVISOR_POLL_SECONDS)

            thread = self._server_thread
            if thread is None or thread.is_alive():
                continue

            if self._stop.is_set():
                return

            log.warning(
                "Prometheus server thread died, attempting restart",
                endpoint=self.endpoint,
            )
            self._shutdown_server()
            try:
                self._bring_up()
            except RuntimeError as e:
                log.error(
                    "Prometheus server restart exhausted retries; giving up",
                    endpoint=self.endpoint, error=str(e),
                )
                return

    def _shutdown_server(self) -> None:
        server, thread = self._server, self._server_thread
        self._server, self._server_thread = None, None

        if server is not None:
            try:
                server.shutdown()
            except BaseException as e:
                log.warning("Prometheus server shutdown raised",
                            endpoint=self.endpoint, error=str(e))
        if thread is not None:
            thread.join(timeout=5.0)
