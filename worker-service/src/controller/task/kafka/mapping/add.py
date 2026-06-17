import worker_service_proto.task as proto

from src.app import AddRequest, AddResponse


def to_add_request(task: proto.Add) -> AddRequest:
    return AddRequest(
        task_id=task.id,
        character_path=task.character_path,
        asset_path=task.asset_path,
        prompt=task.prompt,
        negative_prompt=task.negative_prompt,
    )


def from_add_response(id: str, response: AddResponse) -> proto.AddCompleted:
    return proto.AddCompleted(
        task_id=id,
        character_path=response.character_path,
    )
