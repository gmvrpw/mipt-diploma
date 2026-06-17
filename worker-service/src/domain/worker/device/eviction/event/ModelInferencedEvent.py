from dataclasses import dataclass

from .EvictionStatEvent import EvictionStatEvent


@dataclass
class ModelInferencedEvent(EvictionStatEvent):
    pass
