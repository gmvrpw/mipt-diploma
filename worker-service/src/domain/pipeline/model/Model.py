import abc
from typing import Generic, TypeVar

import structlog
from structlog.stdlib import BoundLogger

from src.config import bound_contextvars
from src.domain.worker import Device
from src.domain.worker.device.eviction.event import ModelInferencedEvent

log: BoundLogger = structlog.get_logger(__name__)

In = TypeVar("In", contravariant=True)

Out = TypeVar("Out", covariant=True)


class Model(abc.ABC, Generic[In, Out]):
    def __init__(self, id: str, gpu_mem: int, cpu_mem: int, device: Device) -> None:
        self._id = id

        self._gpu_mem = gpu_mem
        self._cpu_mem = cpu_mem

        self._device = device

    @property
    def id(self):
        return self._id

    @property
    def gpu_mem(self):
        return self._gpu_mem

    @property
    def cpu_mem(self):
        return self._cpu_mem

    def __call__(self, **inp: In) -> Out:
        with bound_contextvars(model=self._id):
            self._device.load(self)

            log.info("Inferencing...")

            out = self.pipe(**inp)

            log.info("Inferenced")

            self._device.signal(ModelInferencedEvent(self.id))

            return out

    def __enter__(self):
        self._device.load(self)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._device.cache(self)

    def load(self) -> None:
        ...

    def unload(self) -> None:
        ...

    def pipe(self, **kwargs: In) -> Out:
        ...
