from typing import overload

from ..pipeline.tasks.input import TaskInput, PipelineInput, LinkInput, ValueInput


@overload
def resolve(inp: TaskInput, resources: dict[str, str]) -> str | int | bool: ...
@overload
def resolve(inp: None, resources: dict[str, str]) -> None: ...


def resolve(
    inp: TaskInput | None,
    resources: dict[str, str],
) -> str | int | bool | None:
    match inp:
        case None:
            return None
        case PipelineInput(input_id=input_id):
            return resources[f"pipeline.inputs.{input_id}"]
        case LinkInput(task_id=task_id, output_id=output_id):
            return resources[f"{task_id}.outputs.{output_id}"]
        case ValueInput(value=value):
            return value
