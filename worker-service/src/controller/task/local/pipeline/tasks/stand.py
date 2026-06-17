from dataclasses import dataclass

from .task import Task, TaskInputs
from .input import TaskInput


@dataclass(frozen=True)
class StandTaskInputs(TaskInputs):
    character: TaskInput
    prompt: TaskInput
    pose: TaskInput | None
    negative_prompt: TaskInput | None


class StandTask(Task[StandTaskInputs]):
    type = "stand"
    outputs = frozenset({"character"})

    def __init__(self, id: str, name: str, inputs: StandTaskInputs):
        super().__init__(id, name, inputs)
