import worker_service_proto.task as proto

from src.app import RmbgRequest, RmbgResponse


def to_rmbg_request(task: proto.Rmbg) -> RmbgRequest:
    return RmbgRequest(
        task_id=task.id,
        frames_path=task.frames_path,
    )


def from_rmbg_response(id: str, response: RmbgResponse) -> proto.RmbgCompleted:
    return proto.RmbgCompleted(
        task_id=id,
        frames_path=response.frames_path,
    )
