import math
from dataclasses import dataclass
from functools import partial
from typing import Protocol

import structlog
from PIL import Image as PILImage
from PIL.Image import Image
from structlog.stdlib import BoundLogger

from src.domain.Pipe import AsyncPipeOut, Pipe, PipeIn
from src.domain.model import Character, Frames
from src.domain.model.error import InferenceCanceledError
from src.domain.pipeline._cycle_detection import (
    DinoV2CycleDetectionPipeline, OnStep,
)
from src.domain.pipeline._wan_animate_pipeline import Wan22ImageToVideoPipeline
from src.domain.pipeline.model.DinoV2Model import DinoV2Model
from src.domain.worker import Ticket
from src.domain.worker.device.Device import Device


log: BoundLogger = structlog.get_logger(__name__)

MAX_AREA = 720 * 1280
MOD_VALUE = 16  # vae_scale_factor_spatial(8) * patch_size[1](2)

CYCLE_DETECTION_BATCH_SIZE = 16


@dataclass
class AnimatePipelineOutput:
    frames: Frames


class AnimatePipeline(Protocol):
    def pipe(
        self,
        first_frame: Character,
        last_frame: Character | None,
        prompt: str,
        num_frames: int,
        loop: bool | None,
        ticket: Ticket,
    ) -> AsyncPipeOut[AnimatePipelineOutput]: ...


class Wan22AnimatePipeline:
    def __init__(
        self, device: Device, model_dir: str, hf_path: str,
    ) -> None:
        self._wan = Wan22ImageToVideoPipeline.shared(
            device=device, model_dir=model_dir, hf_path=hf_path,
        )
        self._cycle_detection = DinoV2CycleDetectionPipeline(
            device=device,
            model=DinoV2Model(device=device, model_dir=model_dir),
            batch_size=CYCLE_DETECTION_BATCH_SIZE,
            model_dir=model_dir,
        )

    def pipe(
        self,
        first_frame: Character,
        last_frame: Character | None,
        prompt: str,
        num_frames: int,
        loop: bool | None,
        ticket: Ticket,
    ) -> AsyncPipeOut[AnimatePipelineOutput]:
        pipe = Pipe[AnimatePipelineOutput]()
        inp, out = PipeIn(pipe), AsyncPipeOut(pipe)
        ticket.use(partial(self._run, first_frame,
                           last_frame, prompt, num_frames, loop, inp))
        return out

    def _run(
        self,
        first_frame: Character,
        last_frame: Character | None,
        prompt: str,
        num_frames: int,
        loop: bool | None,
        output: PipeIn[AnimatePipelineOutput],
    ) -> None:
        try:
            first_image = first_frame.image.convert("RGB")
            last_image = (
                last_frame.image.convert(
                    "RGB") if last_frame is not None else None
            )

            height, width = _compute_dims(first_image)
            on_step = _cancel_on_closed(output)

            result = self._wan(
                image=first_image,
                prompt=prompt,
                last_image=last_image,
                num_frames=num_frames,
                height=height,
                width=width,
                on_step=on_step,
            )
            assert isinstance(result, list)
            frames = _frames_from_video(result[0])

            if loop:
                frames = self._trim_to_best_cycle(frames, on_step)

            output.pipe(AnimatePipelineOutput(frames=frames))
            output.close()
        except InferenceCanceledError:
            return

    def _trim_to_best_cycle(
        self, frames: Frames, on_step: OnStep,
    ) -> Frames:
        cycles = self._cycle_detection.detect(
            frames, max_cycles=1, on_step=on_step,
        )
        if not cycles:
            log.warning("no cycle detected; returning full frames",
                        num_frames=len(frames))
            return frames
        cycle = cycles[0]
        log.info("trimming frames to best cycle",
                 start=cycle.start, end=cycle.end,
                 period=cycle.period, score=cycle.score)
        return frames[cycle.start:cycle.end + 1]


def _compute_dims(image: Image) -> tuple[int, int]:
    aspect = image.height / image.width
    height = round(math.sqrt(MAX_AREA * aspect)) // MOD_VALUE * MOD_VALUE
    width = round(math.sqrt(MAX_AREA / aspect)) // MOD_VALUE * MOD_VALUE
    return max(height, MOD_VALUE), max(width, MOD_VALUE)


def _frames_from_video(video_frames: list[Image]) -> Frames:
    count = len(video_frames)
    width, height = video_frames[0].size
    atlas = PILImage.new("RGB", (width * count, height))
    for i, frame in enumerate(video_frames):
        atlas.paste(frame, (i * width, 0))
    return Frames(
        atlas=atlas, width=width, height=height, count=count, offset=0,
    )


def _cancel_on_closed(output: PipeIn):
    def check() -> None:
        if output.closed:
            raise InferenceCanceledError()
    return check
