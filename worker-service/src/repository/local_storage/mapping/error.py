import asyncio
import errno

from src.domain.model.error import (
    ServiceError,
    ServicePermissionError,
    ServiceTimeoutError,
    ServiceUnavailableError,
    EntityNotFoundError
)

_UNAVAILABLE_ERRNOS = {
    errno.ENODEV,    # No such device
    errno.ENXIO,     # No such device or address
    errno.EIO,       # I/O error
    errno.EROFS,     # Read-only file system
    errno.ENOSPC,    # No space left on device
    errno.EBUSY,     # Device or resource busy
    errno.ESTALE,    # Stale file handle (NFS)
    errno.EHOSTDOWN if hasattr(errno, "EHOSTDOWN") else -1,
    errno.ENETDOWN if hasattr(errno, "ENETDOWN") else -1,
}


def to_domain_error(e: BaseException):
    if isinstance(e, FileNotFoundError):
        return EntityNotFoundError()
    if isinstance(e, asyncio.TimeoutError):
        return ServiceTimeoutError()
    if isinstance(e, PermissionError):
        return ServicePermissionError()
    if isinstance(e, OSError) and e.errno in _UNAVAILABLE_ERRNOS:
        return ServiceUnavailableError()
    return ServiceError()
