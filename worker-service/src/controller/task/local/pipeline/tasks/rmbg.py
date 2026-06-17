from dataclasses import dataclass

from .task import Task, TaskInputs
from .input import TaskInput


@dataclass(frozen=True)
class RmbgTaskInputs(TaskInputs):
    frames: TaskInput


class RmbgTask(Task[RmbgTaskInputs]):
    type = "rmbg"
    outputs = frozenset({"frames"})

    def __init__(self, id: str, name: str, inputs: RmbgTaskInputs):
        super().__init__(id, name, inputs)
