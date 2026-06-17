from dataclasses import dataclass

from .EvictionStatEvent import EvictionStatEvent


@dataclass
class ModelUnloadedEvent(EvictionStatEvent):
    pass
