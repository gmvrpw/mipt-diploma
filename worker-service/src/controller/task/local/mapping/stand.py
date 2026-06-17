from src.app import StandRequest, StandResponse

from ..pipeline.tasks.stand import StandTask

from .resolve import resolve


def to_stand_request(task: StandTask, resources: dict[str, str]) -> StandRequest:
    pose = resolve(task.inputs.pose, resources)
    negative = resolve(task.inputs.negative_prompt, resources)
    return StandRequest(
        task_id=task.id,
        character_path=str(resolve(task.inputs.character, resources)),
        prompt=str(resolve(task.inputs.prompt, resources)),
        pose_path=str(pose) if pose is not None else None,
        negative_prompt=str(negative) if negative is not None else None,
    )


def from_stand_response(response: StandResponse) -> dict[str, str]:
    return {f"{response.task_id}.outputs.character": response.character_path}
