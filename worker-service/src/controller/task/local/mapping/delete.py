from src.app import DeleteRequest, DeleteResponse

from ..pipeline.tasks.delete import DeleteTask

from .resolve import resolve


def to_delete_request(task: DeleteTask, resources: dict[str, str]) -> DeleteRequest:
    negative = resolve(task.inputs.negative_prompt, resources)
    return DeleteRequest(
        task_id=task.id,
        character_path=str(resolve(task.inputs.character, resources)),
        prompt=str(resolve(task.inputs.prompt, resources)),
        negative_prompt=str(negative) if negative is not None else None,
    )


def from_delete_response(response: DeleteResponse) -> dict[str, str]:
    return {f"{response.task_id}.outputs.character": response.character_path}
