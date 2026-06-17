from .pipeline import PipelineInput
from .link import LinkInput
from .value import ValueInput

TaskInput = PipelineInput | LinkInput | ValueInput


__all__ = ["PipelineInput", "LinkInput", "ValueInput", "TaskInput"]

