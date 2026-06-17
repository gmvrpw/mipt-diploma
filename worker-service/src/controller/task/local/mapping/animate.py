from src.app import AnimateRequest, AnimateResponse

from ..pipeline.tasks.animate import AnimateTask

from .resolve import resolve


_DEFAULT_NUM_FRAMES = 45


def to_animate_request(task: AnimateTask, resources: dict[str, str]) -> AnimateRequest:
    last_frame = resolve(task.inputs.last_frame, resources)
    num_frames = resolve(task.inputs.num_frames, resources)
    loop = resolve(task.inputs.loop, resources)
    return AnimateRequest(
        task_id=task.id,
        first_frame_path=str(resolve(task.inputs.first_frame, resources)),
        prompt=str(resolve(task.inputs.prompt, resources)),
        last_frame_path=str(last_frame) if last_frame is not None else None,
        num_frames=int(num_frames) if num_frames is not None else _DEFAULT_NUM_FRAMES,
        loop=bool(loop) if loop is not None else None,
    )


def from_animate_response(response: AnimateResponse) -> dict[str, str]:
    return {f"{response.task_id}.outputs.frames": response.frames_path}
