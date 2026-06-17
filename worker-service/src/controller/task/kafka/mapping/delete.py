import worker_service_proto.task as proto

from src.app import DeleteRequest, DeleteResponse


def to_delete_request(task: proto.Delete) -> DeleteRequest:
    return DeleteRequest(
        task_id=task.id,
        character_path=task.character_path,
        prompt=task.prompt,
        negative_prompt=task.negative_prompt,
    )


def from_delete_response(id: str, response: DeleteResponse) -> proto.DeleteCompleted:
    return proto.DeleteCompleted(
        task_id=id,
        character_path=response.character_path,
    )
