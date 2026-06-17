import asyncio

from botocore.exceptions import (
    ClientError,
    ConnectTimeoutError,
    EndpointConnectionError,
    ReadTimeoutError,
)

from src.domain.model.error import (
    EntityNotFoundError,
    ServiceError,
    ServicePermissionError,
    ServiceTimeoutError,
    ServiceUnavailableError,
)


_NOT_FOUND_CODES = {"NoSuchKey", "404", "NotFound", "NoSuchBucket"}
_PERMISSION_CODES = {"AccessDenied", "403", "Forbidden", "InvalidAccessKeyId",
                     "SignatureDoesNotMatch"}
_UNAVAILABLE_CODES = {"ServiceUnavailable", "SlowDown", "503"}


def to_domain_error(e: BaseException):
    if isinstance(e, ClientError):
        code = e.response.get("Error", {}).get("Code", "")
        if code in _NOT_FOUND_CODES:
            return EntityNotFoundError()
        if code in _PERMISSION_CODES:
            return ServicePermissionError()
        if code in _UNAVAILABLE_CODES:
            return ServiceUnavailableError()
        return ServiceError()
    if isinstance(e, (ConnectTimeoutError, ReadTimeoutError, asyncio.TimeoutError)):
        return ServiceTimeoutError()
    if isinstance(e, EndpointConnectionError):
        return ServiceUnavailableError()
    return ServiceError()
