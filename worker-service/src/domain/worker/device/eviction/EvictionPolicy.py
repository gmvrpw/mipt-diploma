import abc
from typing import Iterable, Protocol

from .event import EvictionStatEvent


class Model(Protocol):
    id: str


class EvictionPolicy(abc.ABC):
    def choose(self, models: Iterable[Model]) -> Model:
        ...

    def signal(self, event: EvictionStatEvent) -> None:
        ...
