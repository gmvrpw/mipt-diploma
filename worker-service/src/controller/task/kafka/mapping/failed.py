import worker_service_proto.task as proto

from src.domain.model.error import (
    DomainError,
    EntityNotFoundError,
    InferenceError,
    ServiceError,
    ServicePermissionError,
    ServiceTimeoutError,
    ServiceUnavailableError,
)


def domain_error_to_code(e: DomainError) -> str:
    if isinstance(e, EntityNotFoundError):
        return "not_found"
    if isinstance(e, ServicePermissionError):
        return "permission_denied"
    if isinstance(e, ServiceTimeoutError):
        return "timeout"
    if isinstance(e, ServiceUnavailableError):
        return "service_unavailable"
    if isinstance(e, InferenceError):
        return "inference_error"
    if isinstance(e, ServiceError):
        return "service_error"
    return "internal"


def to_failed_response(task_id: str, error_code: str, error_message: str) -> proto.Failed:
    return proto.Failed(
        task_id=task_id,
        error_code=error_code,
        error_message=error_message,
    )
