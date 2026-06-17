from dataclasses import dataclass
from src.domain.model.error.DomainError import DomainError


@dataclass
class InferenceError(DomainError):
    pass
