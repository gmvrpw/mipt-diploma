import worker_service_proto.task as proto

from src.app import AnimateRequest, AnimateResponse


def to_animate_request(task: proto.Animate) -> AnimateRequest:
    return AnimateRequest(
        task_id=task.id,
        first_frame_path=task.first_frame_path,
        prompt=task.prompt,
        last_frame_path=task.last_frame_path,
        num_frames=task.num_frames,
        loop=task.loop,
    )


def from_animate_response(id: str, response: AnimateResponse) -> proto.AnimateCompleted:
    return proto.AnimateCompleted(
        task_id=id,
        frames_path=response.frames_path,
    )
