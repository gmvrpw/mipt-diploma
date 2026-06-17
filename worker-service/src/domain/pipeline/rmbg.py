from dataclasses import dataclass
from functools import partial
from typing import Protocol

import torch
import torch.nn.functional as F
from PIL import Image as PILImage
from PIL.Image import Image

from src.domain.Pipe import AsyncPipeOut, Pipe, PipeIn
from src.domain.model import Frames
from src.domain.model.error import InferenceCanceledError, InferenceError
from src.domain.pipeline.model.InSPyReNetBackboneModel import InSPyReNetBackboneModel
from src.domain.pipeline.model.InSPyReNetDecoderModel import InSPyReNetDecoderModel
from src.domain.pipeline.model.inspyre_net import ImagePyramid, Transition
from src.domain.worker import Ticket
from src.domain.worker.device.Device import Device
from src.domain.worker.device.eviction.event import ModelInferencedEvent


DEVICE = "cuda"
DTYPE = torch.float32

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

LR_BASE_SIZE = 384
HR_THRESHOLD = 512
MAX_HR_SIZE = 1024

ALPHA_THRESHOLD: float | None = None


@dataclass
class RmbgPipelineOutput:
    frames: Frames


class RmbgPipeline(Protocol):
    def pipe(self, frames: Frames, ticket: Ticket) -> AsyncPipeOut[RmbgPipelineOutput]: ...


def _resize_bilinear(x: torch.Tensor, size) -> torch.Tensor:
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


def _round_to_multiple(value: int, multiple: int) -> int:
    return max(multiple, (value // multiple) * multiple)


class InspyreNetRmbgPipeline:
    def __init__(
        self,
        device: Device,
        backbone: InSPyReNetBackboneModel,
        decoder: InSPyReNetDecoderModel,
        batch_size: int,
        hf_path: str,
    ) -> None:
        self._device = device
        self._backbone = backbone
        self._decoder = decoder
        self._batch_size = batch_size
        self._hf_path = hf_path

        self._image_pyramid: ImagePyramid | None = None
        self._transition0: Transition | None = None
        self._transition1: Transition | None = None
        self._transition2: Transition | None = None
        self._mean: torch.Tensor | None = None
        self._std: torch.Tensor | None = None

    def _init_state(self) -> None:
        self._image_pyramid = ImagePyramid(7, 1).to(device=DEVICE, dtype=DTYPE)
        self._transition0 = Transition(17).to(device=DEVICE, dtype=DTYPE)
        self._transition1 = Transition(9).to(device=DEVICE, dtype=DTYPE)
        self._transition2 = Transition(5).to(device=DEVICE, dtype=DTYPE)
        self._mean = torch.tensor(
            IMAGENET_MEAN, device=DEVICE, dtype=DTYPE).view(1, 3, 1, 1)
        self._std = torch.tensor(
            IMAGENET_STD, device=DEVICE, dtype=DTYPE).view(1, 3, 1, 1)

    def _reset_state(self) -> None:
        self._image_pyramid = None
        self._transition0 = None
        self._transition1 = None
        self._transition2 = None
        self._mean = None
        self._std = None
        torch.cuda.empty_cache()
        self._init_state()

    def pipe(
        self,
        frames: Frames,
        ticket: Ticket,
    ) -> AsyncPipeOut[RmbgPipelineOutput]:
        pipe = Pipe()
        inp, out = PipeIn(pipe), AsyncPipeOut(pipe)
        ticket.use(partial(self._run, frames, inp))
        return out

    def _run(
        self,
        frames: Frames,
        output: PipeIn[RmbgPipelineOutput],
    ) -> None:
        self._init_state()
        try:
            rgb_tensor = self._preprocess(frames)
            self._check_cancelled(output)

            saliency = self._infer(rgb_tensor, output)
            self._check_cancelled(output)

            result_frames = self._postprocess(frames, rgb_tensor, saliency)
            output.pipe(RmbgPipelineOutput(frames=result_frames))
            output.close()
        except InferenceCanceledError:
            return
        finally:
            self._reset_state()

    def _check_cancelled(self, output: PipeIn) -> None:
        if output.closed:
            raise InferenceCanceledError()

    def _preprocess(self, frames: Frames) -> torch.Tensor:
        h, w = frames.height, frames.width
        if max(h, w) > MAX_HR_SIZE:
            raise InferenceError(
                f"Frame size {w}x{h} exceeds max supported {MAX_HR_SIZE}x{MAX_HR_SIZE}",
            )

        tensors: list[torch.Tensor] = []
        for i in range(len(frames)):
            frame = frames[i].convert("RGB")
            arr = torch.frombuffer(frame.tobytes(), dtype=torch.uint8).clone()
            arr = arr.view(h, w, 3).permute(
                2, 0, 1).contiguous().float() / 255.0
            tensors.append(arr)
        batch = torch.stack(tensors, dim=0).to(device=DEVICE, dtype=DTYPE)
        return batch

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        assert self._mean is not None and self._std is not None
        return (x - self._mean) / self._std

    def _infer(self, rgb: torch.Tensor, output: PipeIn) -> torch.Tensor:
        N, _, H, W = rgb.shape

        results: list[torch.Tensor] = []
        with self._backbone, self._decoder:
            for start in range(0, N, self._batch_size):
                self._check_cancelled(output)
                chunk = rgb[start:start + self._batch_size]
                pred = self._infer_chunk(chunk, H, W)
                results.append(pred)

        return torch.cat(results, dim=0)

    def _infer_chunk(
        self,
        rgb_chunk: torch.Tensor,
        H: int,
        W: int,
    ) -> torch.Tensor:
        x = self._normalize(rgb_chunk)
        target_hw = self._target_hw(H, W)
        x_full = _resize_bilinear(x, target_hw) if target_hw != (H, W) else x

        if max(H, W) <= HR_THRESHOLD:
            d0 = self._forward_inspyre(x_full, target_hw)["saliency"][-1]
        else:
            x_lr = _resize_bilinear(x, (LR_BASE_SIZE, LR_BASE_SIZE))
            lr_out = self._forward_inspyre(x_lr, (LR_BASE_SIZE, LR_BASE_SIZE))
            hr_out = self._forward_inspyre(x_full, target_hw)
            d0 = self._blend_pyramids(lr_out, hr_out)

        d0 = _resize_bilinear(d0, (H, W))
        return self._normalize_saliency(d0)

    @staticmethod
    def _target_hw(H: int, W: int) -> tuple[int, int]:
        if max(H, W) <= HR_THRESHOLD:
            return (LR_BASE_SIZE, LR_BASE_SIZE)
        return (_round_to_multiple(H, 32), _round_to_multiple(W, 32))

    def _forward_inspyre(
        self,
        x: torch.Tensor,
        target_hw: tuple[int, int],
    ) -> dict[str, list[torch.Tensor]]:
        features = self._backbone.pipe(x=x)
        self._device.signal(ModelInferencedEvent(self._backbone.id))

        out = self._decoder.pipe(features=features, target_hw=target_hw)
        self._device.signal(ModelInferencedEvent(self._decoder.id))
        return out

    def _blend_pyramids(
        self,
        lr_out: dict[str, list[torch.Tensor]],
        hr_out: dict[str, list[torch.Tensor]],
    ) -> torch.Tensor:
        assert self._image_pyramid is not None
        assert self._transition0 is not None
        assert self._transition1 is not None
        assert self._transition2 is not None

        lr_d0 = lr_out["saliency"][-1]
        hr_d3 = hr_out["saliency"][0]
        hr_p2, hr_p1, hr_p0 = hr_out["laplacian"]

        d3 = _resize_bilinear(lr_d0, hr_d3.shape[-2:])

        t2 = _resize_bilinear(self._transition2(d3), hr_p2.shape[-2:])
        p2 = t2 * hr_p2
        d2 = self._image_pyramid.reconstruct(d3, p2)

        t1 = _resize_bilinear(self._transition1(d2), hr_p1.shape[-2:])
        p1 = t1 * hr_p1
        d1 = self._image_pyramid.reconstruct(d2, p1)

        t0 = _resize_bilinear(self._transition0(d1), hr_p0.shape[-2:])
        p0 = t0 * hr_p0
        d0 = self._image_pyramid.reconstruct(d1, p0)
        return d0

    @staticmethod
    def _normalize_saliency(d0: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(d0)
        flat_min = pred.amin(dim=(2, 3), keepdim=True)
        flat_max = pred.amax(dim=(2, 3), keepdim=True)
        return (pred - flat_min) / (flat_max - flat_min + 1e-8)

    def _postprocess(
        self,
        frames: Frames,
        rgb: torch.Tensor,
        saliency: torch.Tensor,
    ) -> Frames:
        if ALPHA_THRESHOLD is not None:
            alpha = (saliency >= ALPHA_THRESHOLD).float()
        else:
            alpha = saliency

        rgb_u8 = (rgb.clamp(0, 1) * 255.0).to(torch.uint8)
        alpha_u8 = (alpha.clamp(0, 1) * 255.0).to(torch.uint8)
        rgba = torch.cat([rgb_u8, alpha_u8], dim=1).cpu()

        N, _, H, W = rgba.shape
        new_atlas = PILImage.new("RGBA", (W * N, H))
        for i in range(N):
            frame_arr = rgba[i].permute(1, 2, 0).contiguous().numpy()
            frame_image: Image = PILImage.fromarray(frame_arr, mode="RGBA")
            new_atlas.paste(frame_image, (i * W, 0))

        return Frames(
            atlas=new_atlas,
            width=W,
            height=H,
            count=N,
            offset=0,
        )
