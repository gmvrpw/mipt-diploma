from src.app import RmbgRequest, RmbgResponse

from ..pipeline.tasks.rmbg import RmbgTask

from .resolve import resolve


def to_rmbg_request(task: RmbgTask, resources: dict[str, str]) -> RmbgRequest:
    return RmbgRequest(
        task_id=task.id,
        frames_path=str(resolve(task.inputs.frames, resources)),
    )


def from_rmbg_response(response: RmbgResponse) -> dict[str, str]:
    return {f"{response.task_id}.outputs.frames": response.frames_path}
