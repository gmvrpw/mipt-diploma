import worker_service_proto.task as proto


def to_cancelled_response(task_id: str) -> proto.Cancelled:
    return proto.Cancelled(task_id=task_id)
