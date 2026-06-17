from __future__ import annotations

import re
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    ValidationError,
)

from .pipeline import Pipeline
from .tasks.add import AddTask, AddTaskInputs
from .tasks.animate import AnimateTask, AnimateTaskInputs
from .tasks.delete import DeleteTask, DeleteTaskInputs
from .tasks.input import LinkInput, PipelineInput, TaskInput, ValueInput
from .tasks.norm_map import NormMapTask, NormMapTaskInputs
from .tasks.rmbg import RmbgTask, RmbgTaskInputs
from .tasks.stand import StandTask, StandTaskInputs
from .tasks.task import Task

_PIPELINE_INPUT_RE = re.compile(r"^pipeline\.inputs\.(.+)$")
_TASK_OUTPUT_RE = re.compile(r"^(.+)\.outputs\.(.+)$")

_anonymous_counter = 0


def _next_anonymous_id() -> str:
    global _anonymous_counter
    _anonymous_counter += 1
    return f"anonymous_{_anonymous_counter}"


def _to_task_input(value: Any) -> Any:
    if isinstance(value, (PipelineInput, LinkInput, ValueInput)):
        return value
    if isinstance(value, str):
        if m := _PIPELINE_INPUT_RE.match(value):
            return PipelineInput(input_id=m.group(1))
        if m := _TASK_OUTPUT_RE.match(value):
            return LinkInput(task_id=m.group(1), output_id=m.group(2))
        return ValueInput(value=value)
    if isinstance(value, (int, bool)):
        return ValueInput(value=value)
    return value


Input = Annotated[TaskInput, BeforeValidator(_to_task_input)]
OptionalInput = Annotated[TaskInput | None, BeforeValidator(_to_task_input)]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)


class _TaskBase(_Strict):
    id: str | None = None
    name: str | None = None


class _AddInputs(_Strict):
    character: Input
    asset: Input
    prompt: Input
    negative_prompt: OptionalInput = None


class _AddSchema(_TaskBase):
    type: Literal["add"]
    inputs: _AddInputs


class _DeleteInputs(_Strict):
    character: Input
    prompt: Input
    negative_prompt: OptionalInput = None


class _DeleteSchema(_TaskBase):
    type: Literal["delete"]
    inputs: _DeleteInputs


class _StandInputs(_Strict):
    character: Input
    prompt: Input
    pose: OptionalInput = None
    negative_prompt: OptionalInput = None


class _StandSchema(_TaskBase):
    type: Literal["stand"]
    inputs: _StandInputs


class _AnimateInputs(_Strict):
    first_frame: Input
    prompt: Input
    last_frame: OptionalInput = None
    num_frames: OptionalInput = None
    loop: OptionalInput = None


class _AnimateSchema(_TaskBase):
    type: Literal["animate"]
    inputs: _AnimateInputs


class _RmbgInputs(_Strict):
    frames: Input


class _RmbgSchema(_TaskBase):
    type: Literal["rmbg"]
    inputs: _RmbgInputs


class _NormMapInputs(_Strict):
    frames: Input


class _NormMapSchema(_TaskBase):
    type: Literal["norm_map"]
    inputs: _NormMapInputs


_TaskSchema = Annotated[
    _AddSchema | _DeleteSchema | _StandSchema | _AnimateSchema | _RmbgSchema | _NormMapSchema,
    Field(discriminator="type"),
]


class _PipelineSchema(_Strict):
    version: Literal["v1"] = "v1"
    name: str = "Anonymous"
    description: str | None = None
    inputs: dict[str, str | None] = Field(default_factory=dict)
    tasks: list[_TaskSchema] = Field(default_factory=list)


def _build_task(schema: Any) -> Task:
    task_id = schema.id or _next_anonymous_id()
    name = schema.name or task_id

    if isinstance(schema, _AddSchema):
        return AddTask(id=task_id, name=name, inputs=AddTaskInputs(
            character=schema.inputs.character,
            asset=schema.inputs.asset,
            prompt=schema.inputs.prompt,
            negative_prompt=schema.inputs.negative_prompt,
        ))
    if isinstance(schema, _DeleteSchema):
        return DeleteTask(id=task_id, name=name, inputs=DeleteTaskInputs(
            character=schema.inputs.character,
            prompt=schema.inputs.prompt,
            negative_prompt=schema.inputs.negative_prompt,
        ))
    if isinstance(schema, _StandSchema):
        return StandTask(id=task_id, name=name, inputs=StandTaskInputs(
            character=schema.inputs.character,
            prompt=schema.inputs.prompt,
            pose=schema.inputs.pose,
            negative_prompt=schema.inputs.negative_prompt,
        ))
    if isinstance(schema, _AnimateSchema):
        return AnimateTask(id=task_id, name=name, inputs=AnimateTaskInputs(
            first_frame=schema.inputs.first_frame,
            prompt=schema.inputs.prompt,
            last_frame=schema.inputs.last_frame,
            num_frames=schema.inputs.num_frames,
            loop=schema.inputs.loop,
        ))
    if isinstance(schema, _RmbgSchema):
        return RmbgTask(id=task_id, name=name, inputs=RmbgTaskInputs(
            frames=schema.inputs.frames,
        ))
    if isinstance(schema, _NormMapSchema):
        return NormMapTask(id=task_id, name=name, inputs=NormMapTaskInputs(
            frames=schema.inputs.frames,
        ))
    raise ValueError(f"Unknown task schema: {type(schema).__name__}")


def _format_errors(error: ValidationError) -> str:
    lines = []
    for err in error.errors():
        loc = ".".join(str(p) for p in err["loc"]) or "<root>"
        lines.append(f"  - {loc}: {err['msg']}")
    return "\n".join(lines)


def _merge_inputs(
    declared: dict[str, str | None],
    specified: dict[str, str],
) -> dict[str, str]:
    inputs: dict[str, str] = dict(specified)
    for input_id, default_value in declared.items():
        if input_id in inputs:
            continue
        if default_value is None:
            raise ValueError(
                f"value not specified for input '{input_id}'"
            )
        inputs[input_id] = default_value
    return inputs


def parse(data: dict, specified_inputs: dict[str, str] = {}) -> Pipeline:
    try:
        schema = _PipelineSchema.model_validate(data)
    except ValidationError as e:
        raise ValueError(
            f"invalid pipeline-v1 config:\n{_format_errors(e)}"
        ) from e

    tasks: dict[str, Task] = {}
    for task_schema in schema.tasks:
        task = _build_task(task_schema)
        if task.id in tasks:
            raise ValueError(f"Duplicate task id: '{task.id}'")
        tasks[task.id] = task

    return Pipeline(
        version="v1",
        name=schema.name,
        description=schema.description,
        inputs=_merge_inputs(schema.inputs, specified_inputs),
        tasks=tasks,
    )
