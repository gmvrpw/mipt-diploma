from dataclasses import dataclass

from .EvictionStatEvent import EvictionStatEvent


@dataclass
class ModelCachedEvent(EvictionStatEvent):
    pass
