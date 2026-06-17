from __future__ import annotations

from typing import Any

from .LokiSink import LokiSink


class LokiProcessor:
    """structlog processor that fans out events to one or more LokiSinks.

    Returns ``event_dict`` unchanged so it continues down the chain to the
    terminal renderer.
    """

    def __init__(self, sinks: list[LokiSink]) -> None:
        self._sinks = sinks

    def __call__(
        self,
        logger: Any,
        method_name: str,
        event_dict: dict[str, Any],
    ) -> dict[str, Any]:
        for sink in self._sinks:
            sink.enqueue(event_dict)
        return event_dict
