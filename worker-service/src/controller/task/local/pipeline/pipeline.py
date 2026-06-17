from dataclasses import dataclass

from .tasks.task import Task


@dataclass
class Pipeline:
    version: str
    name: str
    description: str | None
    inputs: dict[str, str]
    tasks: dict[str, Task]
