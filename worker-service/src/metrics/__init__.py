"""Process-wide Prometheus metric definitions.

Metrics are declared as module-level globals so any layer can import them
without dependency-injection plumbing. ``configure`` resets them under a
namespace and emits ``build_info`` carrying static labels.
"""

from __future__ import annotations

from typing import Any, Final

try:
    from prometheus_client import (
        REGISTRY,
        Counter,
        Gauge,
        Histogram,
    )
    _PROMETHEUS_AVAILABLE = True
except ImportError:
    # Standalone mode does not ship prometheus_client. Substitute no-op stubs
    # that satisfy the metric-call sites in code paths shared with service mode
    # (e.g. Scheduler). configure() is also a no-op in this case.
    _PROMETHEUS_AVAILABLE = False

    class _NoopMetric:
        def __init__(self, *_a: Any, **_kw: Any) -> None: ...
        def labels(self, *_a: Any, **_kw: Any) -> "_NoopMetric": return self
        def set(self, *_a: Any, **_kw: Any) -> None: ...
        def inc(self, *_a: Any, **_kw: Any) -> None: ...
        def observe(self, *_a: Any, **_kw: Any) -> None: ...

    Counter = Gauge = Histogram = _NoopMetric  # type: ignore[misc, assignment]
    REGISTRY = None  # type: ignore[assignment]

_DEFAULT_NAMESPACE: Final[str] = ""

# All globals are rebuilt by configure(). Initial declarations use the
# default registry with no namespace so that any import-time use does not
# crash before configure() runs.

tasks_received_total: Counter
tasks_completed_total: Counter
tasks_failed_total: Counter
tasks_cancelled_total: Counter

ticket_duration_seconds: Histogram
ml_inference_duration_seconds: Histogram
storage_operation_duration_seconds: Histogram

storage_bytes_in_total: Counter
storage_bytes_out_total: Counter

scheduler_queue_size: Gauge

build_info: Gauge


def configure(namespace: str = _DEFAULT_NAMESPACE,
              labels: dict[str, str] | None = None) -> None:
    """Re-declare all metrics under ``namespace`` and emit build_info.

    Safe to call once at startup. Existing collectors are first unregistered
    so the function is idempotent across tests.
    """
    global tasks_received_total, tasks_completed_total, tasks_failed_total
    global tasks_cancelled_total
    global ticket_duration_seconds, ml_inference_duration_seconds
    global storage_operation_duration_seconds
    global storage_bytes_in_total, storage_bytes_out_total
    global scheduler_queue_size, build_info

    _unregister_existing()

    ns = namespace.strip()

    tasks_received_total = Counter(
        "tasks_received_total",
        "Tasks received by the worker controller.",
        labelnames=("task_type",),
        namespace=ns,
    )
    tasks_completed_total = Counter(
        "tasks_completed_total",
        "Tasks finished successfully.",
        labelnames=("task_type",),
        namespace=ns,
    )
    tasks_failed_total = Counter(
        "tasks_failed_total",
        "Tasks that failed with an error.",
        labelnames=("task_type",),
        namespace=ns,
    )
    tasks_cancelled_total = Counter(
        "tasks_cancelled_total",
        "Tasks cancelled before completion.",
        labelnames=("task_type",),
        namespace=ns,
    )

    ticket_duration_seconds = Histogram(
        "ticket_duration_seconds",
        "End-to-end task ticket execution time.",
        labelnames=("task_type",),
        namespace=ns,
    )
    ml_inference_duration_seconds = Histogram(
        "ml_inference_duration_seconds",
        "ML model inference time per model.",
        labelnames=("model",),
        namespace=ns,
    )
    storage_operation_duration_seconds = Histogram(
        "storage_operation_duration_seconds",
        "Storage get/set operation duration.",
        labelnames=("operation",),
        namespace=ns,
    )

    storage_bytes_in_total = Counter(
        "storage_bytes_in_total",
        "Bytes fetched from storage.",
        namespace=ns,
    )
    storage_bytes_out_total = Counter(
        "storage_bytes_out_total",
        "Bytes written to storage.",
        namespace=ns,
    )

    scheduler_queue_size = Gauge(
        "scheduler_queue_size",
        "Current scheduler queue length.",
        namespace=ns,
    )

    build_info_labels = dict(labels or {})
    build_info = Gauge(
        "build_info",
        "Static process metadata as labels.",
        labelnames=tuple(build_info_labels.keys()) or ("none",),
        namespace=ns,
    )
    build_info.labels(**(build_info_labels or {"none": ""})).set(1)


def _unregister_existing() -> None:
    if not _PROMETHEUS_AVAILABLE:
        return
    for name in (
        "tasks_received_total", "tasks_completed_total", "tasks_failed_total",
        "tasks_cancelled_total",
        "ticket_duration_seconds", "ml_inference_duration_seconds",
        "storage_operation_duration_seconds",
        "storage_bytes_in_total", "storage_bytes_out_total",
        "scheduler_queue_size", "build_info",
    ):
        metric = globals().get(name)
        if metric is None:
            continue
        try:
            REGISTRY.unregister(metric)
        except Exception:
            pass


configure()


__all__ = [
    "configure",
    "tasks_received_total", "tasks_completed_total", "tasks_failed_total",
    "tasks_cancelled_total",
    "ticket_duration_seconds", "ml_inference_duration_seconds",
    "storage_operation_duration_seconds",
    "storage_bytes_in_total", "storage_bytes_out_total",
    "scheduler_queue_size", "build_info",
]
