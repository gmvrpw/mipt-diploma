from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any, Protocol, cast

import torch
import torch.nn.functional as F
from PIL.Image import Image

from src.domain.Pipe import AsyncPipeOut, Pipe, PipeIn
from src.domain.model import Frames
from src.domain.model.error import InferenceCanceledError
from src.domain.pipeline.model.DinoV2Model import CHECKPOINT, DinoV2Model
from src.domain.worker import Ticket
from src.domain.worker.device.Device import Device
from src.domain.worker.device.eviction.event import ModelInferencedEvent

from transformers import AutoImageProcessor


OnStep = Callable[[], None]


def _fire(on_step: OnStep | None) -> None:
    if on_step is not None:
        on_step()


def _cancel_on_closed(output: PipeIn[Any]) -> OnStep:
    def check() -> None:
        if output.closed:
            raise InferenceCanceledError()
    return check


DEVICE = "cuda"

DEFAULT_MIN_PERIOD = 8
DEFAULT_MIN_STITCH_SIMILARITY = 0.93
DEFAULT_MIN_INTRA_CYCLE_DIFF = 0.05
DEFAULT_SECOND_ORDER_WEIGHT = 0.5

SCORE_WEIGHT_STITCH = 1.0
SCORE_WEIGHT_MOTION = 0.5
SCORE_WEIGHT_SMOOTHNESS = 0.3


@dataclass
class Cycle:
    start: int
    end: int
    period: int
    score: float


@dataclass
class CycleDetectionPipelineOutput:
    cycles: list[Cycle]


class CycleDetectionPipeline(Protocol):
    def pipe(
        self,
        frames: Frames,
        max_cycles: int,
        ticket: Ticket,
    ) -> AsyncPipeOut[CycleDetectionPipelineOutput]: ...


class DinoV2CycleDetectionPipeline:
    def __init__(
        self,
        device: Device,
        model: DinoV2Model,
        batch_size: int,
        model_dir: str,
        min_period: int = DEFAULT_MIN_PERIOD,
        min_stitch_similarity: float = DEFAULT_MIN_STITCH_SIMILARITY,
        min_intra_cycle_diff: float = DEFAULT_MIN_INTRA_CYCLE_DIFF,
    ) -> None:
        self._device = device
        self._model = model
        self._batch_size = batch_size
        self._min_period = min_period
        self._min_stitch_similarity = min_stitch_similarity
        self._min_intra_cycle_diff = min_intra_cycle_diff

        self._processor: Any = AutoImageProcessor.from_pretrained(
            CHECKPOINT, cache_dir=model_dir,
        )

    def detect(
        self,
        frames: Frames,
        max_cycles: int,
        on_step: OnStep | None = None,
    ) -> list[Cycle]:
        embeddings = self._extract_embeddings(frames, on_step)
        sim = self._similarity_matrix(embeddings)
        _fire(on_step)
        return _detect_pair_cycles(
            sim,
            min_period=self._min_period,
            min_stitch_similarity=self._min_stitch_similarity,
            min_intra_cycle_diff=self._min_intra_cycle_diff,
            max_cycles=max_cycles,
        )

    def pipe(
        self,
        frames: Frames,
        max_cycles: int,
        ticket: Ticket,
    ) -> AsyncPipeOut[CycleDetectionPipelineOutput]:
        pipe = Pipe[CycleDetectionPipelineOutput]()
        inp, out = PipeIn(pipe), AsyncPipeOut(pipe)
        ticket.use(partial(self._run, frames, max_cycles, inp))
        return out

    def _run(
        self,
        frames: Frames,
        max_cycles: int,
        output: PipeIn[CycleDetectionPipelineOutput],
    ) -> None:
        try:
            cycles = self.detect(
                frames, max_cycles, on_step=_cancel_on_closed(output),
            )
            output.pipe(CycleDetectionPipelineOutput(cycles=cycles))
            output.close()
        except InferenceCanceledError:
            return

    def _extract_embeddings(
        self,
        frames: Frames,
        on_step: OnStep | None,
    ) -> torch.Tensor:
        n = len(frames)
        chunks: list[torch.Tensor] = []
        with self._model:
            for start in range(0, n, self._batch_size):
                _fire(on_step)
                end = min(start + self._batch_size, n)
                batch_images: list[Image] = [
                    frames[i].convert("RGB") for i in range(start, end)
                ]
                pixel_values = cast(
                    torch.Tensor,
                    self._processor(
                        batch_images, return_tensors="pt",
                    )["pixel_values"],
                )
                emb = self._model.pipe(pixel_values=pixel_values)
                self._device.signal(ModelInferencedEvent(self._model.id))
                emb = F.normalize(emb, dim=-1)
                chunks.append(emb)
        return torch.cat(chunks, dim=0)

    @staticmethod
    def _similarity_matrix(embeddings: torch.Tensor) -> torch.Tensor:
        return (embeddings @ embeddings.T).cpu()


def _intra_cycle_diff(sim: torch.Tensor, start: int, end: int) -> float:
    sub = sim[start:end + 1, start:end + 1]
    return 1.0 - float(sub.min())


def _stitch_smoothness(
    sim: torch.Tensor, start: int, end: int,
    second_order_weight: float = DEFAULT_SECOND_ORDER_WEIGHT,
) -> float:
    n = sim.shape[0]
    if end - start < 3 or end >= n:
        return 0.0

    inside = sim[
        torch.arange(start, end - 1),
        torch.arange(start + 1, end),
    ]
    mu = float(inside.mean())

    stitch_step = float(sim[end - 1, start])
    smoothness_1 = 1.0 - abs(stitch_step - mu)

    a = float(sim[end - 2, end - 1])
    b = stitch_step
    c = float(sim[start, start + 1])
    smoothness_2 = 1.0 - max(abs(b - a), abs(b - c))

    total = smoothness_1 + second_order_weight * smoothness_2
    return total / (1.0 + second_order_weight)


def _detect_pair_cycles(
    sim: torch.Tensor,
    *,
    min_period: int,
    min_stitch_similarity: float,
    min_intra_cycle_diff: float,
    max_cycles: int,
) -> list[Cycle]:
    if max_cycles <= 0:
        return []

    n = sim.shape[0]
    raw: list[tuple[int, int, float, float, float]] = []
    for i in range(n):
        for j in range(i + min_period, n):
            stitch = float(sim[i, j])
            if stitch < min_stitch_similarity:
                continue
            motion = _intra_cycle_diff(sim, i, j)
            if motion < min_intra_cycle_diff:
                continue
            smooth = _stitch_smoothness(sim, i, j)
            raw.append((i, j, stitch, motion, smooth))

    raw.sort(key=lambda x: -_score(x[2], x[3], x[4]))

    cycles: list[Cycle] = []
    for i, j, stitch, motion, smooth in raw[:max_cycles]:
        cycles.append(Cycle(
            start=i, end=j, period=j - i,
            score=_score(stitch, motion, smooth),
        ))
    return cycles


def _score(stitch: float, motion: float, smooth: float) -> float:
    return (
        SCORE_WEIGHT_STITCH * stitch
        + SCORE_WEIGHT_MOTION * motion
        + SCORE_WEIGHT_SMOOTHNESS * smooth
    )
