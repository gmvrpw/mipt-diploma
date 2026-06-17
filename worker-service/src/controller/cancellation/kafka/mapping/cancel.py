import worker_service_proto.task as proto

from src.app import CancelTaskRequest


def to_cancel_task_request(request: proto.Cancelled):
    return CancelTaskRequest(task_id=request.task_id)
