from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineInput:
    input_id: str