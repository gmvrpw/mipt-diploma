from dataclasses import dataclass

from .task import Task, TaskInputs
from .input import TaskInput


@dataclass(frozen=True)
class AnimateTaskInputs(TaskInputs):
    first_frame: TaskInput
    prompt: TaskInput
    last_frame: TaskInput | None
    num_frames: TaskInput | None
    loop: TaskInput | None


class AnimateTask(Task[AnimateTaskInputs]):
    type = "animate"
    outputs = frozenset({"frames"})

    def __init__(self, id: str, name: str, inputs: AnimateTaskInputs):
        super().__init__(id, name, inputs)
