from dataclasses import dataclass


@dataclass(frozen=True)
class ValueInput:
    value: str | int | bool