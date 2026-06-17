from dataclasses import dataclass


@dataclass(frozen=True)
class LinkInput:
    task_id: str
    output_id: str