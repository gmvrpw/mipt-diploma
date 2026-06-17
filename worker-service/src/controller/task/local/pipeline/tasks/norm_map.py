from dataclasses import dataclass

from .task import Task, TaskInputs
from .input import TaskInput


@dataclass(frozen=True)
class NormMapTaskInputs(TaskInputs):
    frames: TaskInput


class NormMapTask(Task[NormMapTaskInputs]):
    type = "norm_map"
    outputs = frozenset({"frames"})

    def __init__(self, id: str, name: str, inputs: NormMapTaskInputs):
        super().__init__(id, name, inputs)
