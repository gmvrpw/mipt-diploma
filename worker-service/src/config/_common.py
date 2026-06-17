from __future__ import annotations

import re
from typing import Annotated, Any

from pydantic import BaseModel, BeforeValidator, ConfigDict, ValidationError


_DURATION_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(ms|s|m|h)\s*$")
_DURATION_UNITS = {"ms": 1e-3, "s": 1.0, "m": 60.0, "h": 3600.0}


def _parse_duration(value: Any) -> Any:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if not isinstance(value, str):
        return value
    m = _DURATION_RE.match(value)
    if m is None:
        raise ValueError(
            f"invalid duration {value!r}; "
            "expected '<number><ms|s|m|h>', e.g. '500ms', '1s', '2m'"
        )
    return float(m.group(1)) * _DURATION_UNITS[m.group(2)]


Duration = Annotated[float, BeforeValidator(_parse_duration)]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


def format_errors(error: ValidationError) -> str:
    lines = []
    for err in error.errors():
        loc = ".".join(str(p) for p in err["loc"]) or "<root>"
        lines.append(f"  - {loc}: {err['msg']}")
    return "\n".join(lines)
