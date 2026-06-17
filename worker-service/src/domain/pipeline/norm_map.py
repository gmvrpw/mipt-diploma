from dataclasses import dataclass
from functools import partial
from typing import Protocol

import torch
from PIL import Image as PILImage
from PIL.Image import Image

from src.domain.Pipe import AsyncPipeOut, Pipe, PipeIn
from src.domain.model import Frames
from src.domain.model.error import InferenceCanceledError, InferenceError
from src.domain.pipeline.model.StableNormalModel import StableNormalModel
from src.domain.worker import Ticket
from src.domain.worker.device.Device import Device
from src.domain.worker.device.eviction.event import ModelInferencedEvent


MAX_SIDE = 1024
MIN_SIDE = 64
SIZE_MULTIPLE = 64

NUM_INFERENCE_STEPS = 10

DEFAULT_NORMAL_RGB = (128, 128, 255)


@dataclass
class NormMapPipelineOutput:
    frames: Frames


class NormMapPipeline(Protocol):
    def pipe(self, frames: Frames, ticket: Ticket) -> AsyncPipeOut[NormMapPipelineOutput]: ...


def _round_to_multiple(value: int, multiple: int) -> int:
    return max(multiple, (value // multiple) * multiple)


def _target_hw(h: int, w: int) -> tuple[int, int]:
    longest = max(h, w)
    if longest > MAX_SIDE:
        scale = MAX_SIDE / longest
        h = int(round(h * scale))
        w = int(round(w * scale))
    return _round_to_multiple(h, SIZE_MULTIPLE), _round_to_multiple(w, SIZE_MULTIPLE)


class StableNormal01NormalMapPipeline:
    def __init__(
        self,
        device: Device,
        model: StableNormalModel,
        batch_size: int,
        hf_path: str,
        num_inference_steps: int = NUM_INFERENCE_STEPS,
    ) -> None:
        self._device = device
        self._model = model
        self._batch_size = batch_size
        self._hf_path = hf_path
        self._num_inference_steps = num_inference_steps

    def pipe(
        self,
        frames: Frames,
        ticket: Ticket,
    ) -> AsyncPipeOut[NormMapPipelineOutput]:
        pipe = Pipe()
        inp, out = PipeIn(pipe), AsyncPipeOut(pipe)
        ticket.use(partial(self._run, frames, inp))
        return out

    def _run(
        self,
        frames: Frames,
        output: PipeIn[NormMapPipelineOutput],
    ) -> None:
        try:
            self._validate(frames)
            self._check_cancelled(output)

            rgb_inputs, alpha_masks = self._preprocess(frames)
            self._check_cancelled(output)

            normals = self._infer(rgb_inputs, output)
            self._check_cancelled(output)

            result = self._postprocess(frames, normals, alpha_masks)
            output.pipe(NormMapPipelineOutput(frames=result))
            output.close()
        except InferenceCanceledError:
            return

    def _check_cancelled(self, output: PipeIn) -> None:
        if output.closed:
            raise InferenceCanceledError()

    def _validate(self, frames: Frames) -> None:
        if frames.height < MIN_SIDE or frames.width < MIN_SIDE:
            raise InferenceError(
                f"Frame size {frames.width}x{frames.height} below minimum "
                f"{MIN_SIDE}x{MIN_SIDE}",
            )

    def _preprocess(
        self,
        frames: Frames,
    ) -> tuple[list[Image], list[Image | None]]:
        target_h, target_w = _target_hw(frames.height, frames.width)

        rgb_inputs: list[Image] = []
        alpha_masks: list[Image | None] = []

        for i in range(len(frames)):
            frame = frames[i]
            if frame.mode == "RGBA":
                alpha = frame.split()[-1]
                white_bg = PILImage.new("RGB", frame.size, (255, 255, 255))
                rgb = PILImage.composite(
                    frame.convert("RGB"), white_bg, alpha,
                )
                alpha_masks.append(alpha)
            else:
                rgb = frame.convert("RGB")
                alpha_masks.append(None)

            if rgb.size != (target_w, target_h):
                rgb = rgb.resize((target_w, target_h), PILImage.Resampling.LANCZOS)
            rgb_inputs.append(rgb)

        return rgb_inputs, alpha_masks

    def _infer(self, rgb_inputs: list[Image], output: PipeIn) -> torch.Tensor:
        results: list[torch.Tensor] = []
        with self._model:
            for start in range(0, len(rgb_inputs), self._batch_size):
                self._check_cancelled(output)
                chunk = rgb_inputs[start:start + self._batch_size]
                pred = self._model.pipe(
                    image=chunk,
                    num_inference_steps=self._num_inference_steps,
                )
                self._device.signal(ModelInferencedEvent(self._model.id))
                results.append(pred.detach().to("cpu", torch.float32))

        return torch.cat(results, dim=0)

    def _postprocess(
        self,
        frames: Frames,
        normals: torch.Tensor,
        alpha_masks: list[Image | None],
    ) -> Frames:
        H, W = frames.height, frames.width
        N = len(frames)

        normals = normals.clamp(-1, 1)
        normals_uint8 = ((normals + 1.0) * 127.5).round().to(torch.uint8)

        new_atlas = PILImage.new("RGBA", (W * N, H))

        for i in range(N):
            frame_arr = normals_uint8[i].permute(1, 2, 0).contiguous().numpy()
            normal_image = PILImage.fromarray(frame_arr, mode="RGB")

            if normal_image.size != (W, H):
                normal_image = normal_image.resize(
                    (W, H), PILImage.Resampling.LANCZOS,
                )

            alpha = alpha_masks[i]
            if alpha is None:
                rgba_image = normal_image.convert("RGBA")
            else:
                if alpha.size != (W, H):
                    alpha = alpha.resize((W, H), PILImage.Resampling.NEAREST)
                background = PILImage.new("RGB", (W, H), DEFAULT_NORMAL_RGB)
                composed = PILImage.composite(normal_image, background, alpha)
                rgba_image = composed.convert("RGBA")
                rgba_image.putalpha(alpha)

            new_atlas.paste(rgba_image, (i * W, 0))

        return Frames(
            atlas=new_atlas,
            width=W,
            height=H,
            count=N,
            offset=0,
        )
