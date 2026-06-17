from src.app import AddRequest, AddResponse

from ..pipeline.tasks.add import AddTask

from .resolve import resolve


def to_add_request(task: AddTask, resources: dict[str, str]) -> AddRequest:
    negative = resolve(task.inputs.negative_prompt, resources)
    return AddRequest(
        task_id=task.id,
        character_path=str(resolve(task.inputs.character, resources)),
        asset_path=str(resolve(task.inputs.asset, resources)),
        prompt=str(resolve(task.inputs.prompt, resources)),
        negative_prompt=str(negative) if negative is not None else None,
    )


def from_add_response(response: AddResponse) -> dict[str, str]:
    return {f"{response.task_id}.outputs.character": response.character_path}
