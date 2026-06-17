from src.app import NormMapRequest, NormMapResponse

from ..pipeline.tasks.norm_map import NormMapTask

from .resolve import resolve


def to_norm_map_request(task: NormMapTask, resources: dict[str, str]) -> NormMapRequest:
    return NormMapRequest(
        task_id=task.id,
        frames_path=str(resolve(task.inputs.frames, resources)),
    )


def from_norm_map_response(response: NormMapResponse) -> dict[str, str]:
    return {f"{response.task_id}.outputs.frames": response.frames_path}
