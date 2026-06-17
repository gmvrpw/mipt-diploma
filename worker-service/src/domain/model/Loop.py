from dataclasses import dataclass


@dataclass
class Loop:
    first_frame_index: int
    last_frame_index: int
    confidence: float


class LoopCollection:
    best: Loop

    def __len__(self) -> int:
        pass
