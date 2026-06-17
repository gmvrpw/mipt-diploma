from dataclasses import dataclass

from .EvictionStatEvent import EvictionStatEvent


@dataclass
class ModelActivatedEvent(EvictionStatEvent):
    pass
