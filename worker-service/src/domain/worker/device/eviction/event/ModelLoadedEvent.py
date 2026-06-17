from dataclasses import dataclass

from .EvictionStatEvent import EvictionStatEvent


@dataclass
class ModelLoadedEvent(EvictionStatEvent):
    pass
