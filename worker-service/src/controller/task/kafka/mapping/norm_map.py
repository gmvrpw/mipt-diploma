import worker_service_proto.task as proto

from src.app import NormMapRequest, NormMapResponse


def to_norm_map_request(task: proto.NormMap) -> NormMapRequest:
    return NormMapRequest(
        task_id=task.id,
        frames_path=task.frames_path,
    )


def from_norm_map_response(id: str, response: NormMapResponse) -> proto.NormMapCompleted:
    return proto.NormMapCompleted(
        task_id=id,
        frames_path=response.frames_path,
    )
