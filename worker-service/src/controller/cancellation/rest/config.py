from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class RestCancellationCertificate:
    cert: str
    key: str


@dataclass(frozen=True)
class RestCancellationAuthorization:
    header: str | None = None
    certificate: RestCancellationCertificate | None = None


@dataclass(frozen=True)
class RestCancellationConfig:
    path: str
    method: Literal["GET", "POST"] = "GET"
    authorization: RestCancellationAuthorization | None = None
    timeout_seconds: float = 5.0
    verify_ssl: bool = True
