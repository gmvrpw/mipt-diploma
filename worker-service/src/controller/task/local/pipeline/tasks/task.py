from dataclasses import dataclass, fields
from typing import Generic, Iterator, TypeVar

from .input import TaskInput


@dataclass(frozen=True)
class TaskInputs:
    def __iter__(self) -> Iterator[TaskInput]:
        return iter(
            getattr(self, f.name) for f in fields(self)
            if getattr(self, f.name) is not None
        )


T = TypeVar('T', bound=TaskInputs)


class Task(Generic[T]):
    type: str = "unknown"
    outputs: frozenset[str] = frozenset()

    def __init__(self, id: str, name: str, inputs: T):
        self._id = id
        self._name = name
        self._inputs = inputs

    @property
    def id(self) -> str:
        return self._id

    @property
    def name(self) -> str:
        return self._name

    @property
    def inputs(self) -> T:
        return self._inputs

