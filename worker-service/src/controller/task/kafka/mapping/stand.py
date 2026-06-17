import worker_service_proto.task as proto

from src.app import StandRequest, StandResponse


def to_stand_request(task: proto.Stand) -> StandRequest:
    return StandRequest(
        task_id=task.id,
        character_path=task.character_path,
        prompt=task.prompt,
        pose_path=task.pose_path,
        negative_prompt=task.negative_prompt,
    )


def from_stand_response(id: str, response: StandResponse) -> proto.StandCompleted:
    return proto.StandCompleted(
        task_id=id,
        character_path=response.character_path,
    )
