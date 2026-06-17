from dataclasses import dataclass

from .task import Task, TaskInputs
from .input import TaskInput


@dataclass(frozen=True)
class AddTaskInputs(TaskInputs):
    character: TaskInput
    asset: TaskInput
    prompt: TaskInput
    negative_prompt: TaskInput | None


class AddTask(Task[AddTaskInputs]):
    type = "add"
    outputs = frozenset({"character"})

    def __init__(self, id: str, name: str, inputs: AddTaskInputs):
        super().__init__(id, name, inputs)
