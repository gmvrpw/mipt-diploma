from typing import Protocol

import structlog
from structlog.contextvars import bound_contextvars
from structlog.stdlib import BoundLogger

from src.domain.worker.device.eviction import EvictionPolicy
from src.domain.worker.device.eviction.event import EvictionStatEvent, \
    ModelLoadedEvent, ModelUnloadedEvent, ModelActivatedEvent, ModelCachedEvent


log: BoundLogger = structlog.get_logger(__name__)


class Model(Protocol):
    @property
    def id(self) -> str:
        ...

    @property
    def gpu_mem(self) -> int:
        ...

    @property
    def cpu_mem(self) -> int:
        ...

    def load(self) -> None:
        ...

    def unload(self) -> None:
        ...


class InsufficientDeviceMemoryError(Exception):
    pass


class Device:
    def __init__(self, eviction_policy: EvictionPolicy, gpu_mem: int) -> None:
        self._eviction_policy = eviction_policy

        self._gpu_mem_limit = gpu_mem

        self._active_models = set[Model]()
        self._cached_models = set[Model]()

    @property
    def _used_gpu_mem(self) -> int:
        return sum(
            m.gpu_mem for m in self._active_models | self._cached_models
        )

    @property
    def _free_gpu_mem(self) -> int:
        return self._gpu_mem_limit - self._used_gpu_mem

    def load(self, model: Model):
        with bound_contextvars(model=model.id):
            if model in self._active_models:
                return

            if model in self._cached_models:
                log.info("Activating cached model...")
                self._cached_models.discard(model)
                self._active_models.add(model)
                self._eviction_policy.signal(ModelActivatedEvent(model.id))
                log.info("Model activated")
                return

            log.info("Loading model...")

            self.free(cpu_mem=model.cpu_mem,
                      gpu_mem=model.gpu_mem)

            model.load()

            self._active_models.add(model)

            self._eviction_policy.signal(ModelLoadedEvent(model.id))
            self._eviction_policy.signal(ModelActivatedEvent(model.id))
            log.info("Model loaded")

    def cache(self, model: Model):
        with bound_contextvars(model=model.id):
            if model not in self._active_models:
                return

            log.info("Caching model...")

            self._active_models.discard(model)
            self._cached_models.add(model)

            self._eviction_policy.signal(ModelCachedEvent(model.id))
            log.info("Model cached")

    def cache_all(self):
        active = list(self._active_models)
        with bound_contextvars(num_models=len(active)):
            log.info("Caching models...")

            for model in active:
                self._active_models.discard(model)
                self._cached_models.add(model)

                self._eviction_policy.signal(ModelCachedEvent(model.id))

            log.info("Models cached")

    def unload(self, model: Model):
        with bound_contextvars(model=model.id):
            if model not in self._active_models and model not in self._cached_models:
                return

            log.info("Unloading model...")

            self._active_models.discard(model)
            self._cached_models.discard(model)
            model.unload()

            self._eviction_policy.signal(ModelUnloadedEvent(model.id))
            log.info("Model unloaded")

    def free(self, cpu_mem: int, gpu_mem: int):
        del cpu_mem

        while self._free_gpu_mem < gpu_mem:
            if not self._cached_models:
                raise InsufficientDeviceMemoryError(
                    f"need {gpu_mem} B free on GPU; "
                    f"{self._used_gpu_mem}/{self._gpu_mem_limit} B in use "
                    f"and no cached models left to evict"
                )

            victim = self._eviction_policy.choose(self._cached_models)
            if victim not in self._cached_models:
                raise InsufficientDeviceMemoryError(
                    f"eviction policy returned {victim.id}, which is not cached"
                )

            self.unload(victim)

    def signal(self, event: EvictionStatEvent):
        self._eviction_policy.signal(event)
