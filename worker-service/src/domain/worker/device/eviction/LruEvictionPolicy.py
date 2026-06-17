from collections import OrderedDict
from typing import Iterable

from .EvictionPolicy import EvictionPolicy, Model
from .event import EvictionStatEvent, ModelInferencedEvent, ModelUnloadedEvent


class NoEvictionCandidateError(Exception):
    pass


class LruEvictionPolicy(EvictionPolicy):
    def __init__(self):
        self._order: OrderedDict[str, None] = OrderedDict()

    def choose(self, models: Iterable[Model]) -> Model:
        candidates = {m.id: m for m in models}
        if not candidates:
            raise NoEvictionCandidateError(
                "eviction requested but no candidate models provided"
            )

        # Never-inferenced candidates are treated as least-recently-used.
        for id, model in candidates.items():
            if id not in self._order:
                return model

        for id in self._order:
            if id in candidates:
                self._order.pop(id, None)
                return candidates[id]

        raise NoEvictionCandidateError(
            "no candidate could be matched to LRU state"
        )

    def signal(self, event: EvictionStatEvent) -> None:
        match event:
            case ModelInferencedEvent(model_id=id):
                self._order.pop(id, None)
                self._order[id] = None
            case ModelUnloadedEvent(model_id=id):
                self._order.pop(id, None)
