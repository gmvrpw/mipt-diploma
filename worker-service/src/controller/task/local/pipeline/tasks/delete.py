from dataclasses import dataclass

from .task import Task, TaskInputs
from .input import TaskInput


@dataclass(frozen=True)
class DeleteTaskInputs(TaskInputs):
    character: TaskInput
    prompt: TaskInput
    negative_prompt: TaskInput | None


class DeleteTask(Task[DeleteTaskInputs]):
    type = "delete"
    outputs = frozenset({"character"})

    def __init__(self, id: str, name: str, inputs: DeleteTaskInputs):
        super().__init__(id, name, inputs)
