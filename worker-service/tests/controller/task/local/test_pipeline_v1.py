import pytest

from src.controller.task.local.pipeline.tasks import (
    AddTask,
    AnimateTask,
    DeleteTask,
    LinkInput,
    NormMapTask,
    PipelineInput,
    RmbgTask,
    StandTask,
    ValueInput,
)
from src.controller.task.local.pipeline.v1 import parse


def test_parses_all_six_task_types():
    data = {
        "version": "v1",
        "inputs": {"src": "/x/a.png", "asset": "/x/b.png"},
        "tasks": [
            {"id": "t1", "type": "add", "inputs": {
                "character": "pipeline.inputs.src",
                "asset": "pipeline.inputs.asset",
                "prompt": "Add hat",
            }},
            {"id": "t2", "type": "delete", "inputs": {
                "character": "t1.outputs.character",
                "prompt": "Remove items",
                "negative_prompt": "low quality",
            }},
            {"id": "t3", "type": "stand", "inputs": {
                "character": "t2.outputs.character",
                "prompt": "Standing pose",
            }},
            {"id": "t4", "type": "animate", "inputs": {
                "first_frame": "t3.outputs.character",
                "prompt": "walking",
                "num_frames": 30,
                "loop": True,
            }},
            {"id": "t5", "type": "rmbg", "inputs": {"frames": "t4.outputs.frames"}},
            {"id": "t6", "type": "norm_map", "inputs": {"frames": "t5.outputs.frames"}},
        ],
    }
    pipeline = parse(data, {})

    assert isinstance(pipeline.tasks["t1"], AddTask)
    assert isinstance(pipeline.tasks["t2"], DeleteTask)
    assert isinstance(pipeline.tasks["t3"], StandTask)
    assert isinstance(pipeline.tasks["t4"], AnimateTask)
    assert isinstance(pipeline.tasks["t5"], RmbgTask)
    assert isinstance(pipeline.tasks["t6"], NormMapTask)

    t1 = pipeline.tasks["t1"]
    assert isinstance(t1.inputs.character, PipelineInput)
    assert t1.inputs.character.input_id == "src"
    assert isinstance(t1.inputs.prompt, ValueInput)

    t2 = pipeline.tasks["t2"]
    assert isinstance(t2.inputs.character, LinkInput)
    assert t2.inputs.character.task_id == "t1"
    assert t2.inputs.character.output_id == "character"

    t4 = pipeline.tasks["t4"]
    assert isinstance(t4.inputs.num_frames, ValueInput)
    assert t4.inputs.num_frames.value == 30
    assert isinstance(t4.inputs.loop, ValueInput)
    assert t4.inputs.loop.value is True


def test_optional_inputs_omitted_become_none():
    data = {
        "version": "v1",
        "tasks": [
            {"id": "t", "type": "stand", "inputs": {
                "character": "x.png",
                "prompt": "pose",
            }},
        ],
    }
    pipeline = parse(data, {})
    task = pipeline.tasks["t"]
    assert task.inputs.pose is None
    assert task.inputs.negative_prompt is None
    # Iter should skip None inputs
    assert all(i is not None for i in task.inputs)


def test_specified_inputs_override_defaults():
    data = {
        "version": "v1",
        "inputs": {"src": "default.png"},
        "tasks": [
            {"id": "t", "type": "rmbg", "inputs": {"frames": "pipeline.inputs.src"}},
        ],
    }
    pipeline = parse(data, {"src": "override.png"})
    assert pipeline.inputs["src"] == "override.png"


def test_undeclared_input_required():
    data = {
        "version": "v1",
        "inputs": {"src": None},
        "tasks": [],
    }
    with pytest.raises(ValueError) as exc:
        parse(data, {})
    assert "src" in str(exc.value)


def test_missing_required_field_is_a_pydantic_error():
    data = {
        "version": "v1",
        "tasks": [{"id": "t", "type": "add", "inputs": {"character": "x"}}],
    }
    with pytest.raises(ValueError) as exc:
        parse(data, {})
    assert "asset" in str(exc.value) and "Field required" in str(exc.value)
