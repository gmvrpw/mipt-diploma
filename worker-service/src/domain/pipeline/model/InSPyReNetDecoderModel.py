import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

from src.domain.pipeline.model.Model import Model
from src.domain.pipeline.model.inspyre_net import ImagePyramid, PAA_d, PAA_e, SICA
from src.domain.worker.device.Device import Device


GB = 1024 ** 3
MB = 1024 ** 2
WEIGHTS_GPU_MEM = 256 * MB
ACTIVATIONS_GPU_MEM = 768 * MB
GPU_MEM = WEIGHTS_GPU_MEM + ACTIVATIONS_GPU_MEM
CPU_MEM = 256 * MB

REPO_ID = "plemeri/InSPyReNet"
CHECKPOINT_FILENAME = "snapshots/Plus_Ultra/latest.pth"

DEVICE = "cuda"
DTYPE = torch.float32

DEPTH = 64
IN_CHANNELS = (128, 128, 256, 512, 1024)
DECODER_PREFIXES = (
    "context1.", "context2.", "context3.", "context4.", "context5.",
    "decoder.",
    "attention0.", "attention1.", "attention2.",
)


class _Decoder(nn.Module):
    def __init__(self, in_channels, depth: int, base_size: list[int]) -> None:
        super().__init__()
        self.context1 = PAA_e(in_channels[0], depth, base_size=base_size, stage=0)
        self.context2 = PAA_e(in_channels[1], depth, base_size=base_size, stage=1)
        self.context3 = PAA_e(in_channels[2], depth, base_size=base_size, stage=2)
        self.context4 = PAA_e(in_channels[3], depth, base_size=base_size, stage=3)
        self.context5 = PAA_e(in_channels[4], depth, base_size=base_size, stage=4)

        self.decoder = PAA_d(depth * 3, depth=depth, base_size=base_size, stage=2)

        self.attention0 = SICA(depth, depth=depth, base_size=base_size, stage=0, lmap_in=True)
        self.attention1 = SICA(depth * 2, depth=depth, base_size=base_size, stage=1, lmap_in=True)
        self.attention2 = SICA(depth * 2, depth=depth, base_size=base_size, stage=2)


def _resize(x: torch.Tensor, size) -> torch.Tensor:
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class InSPyReNetDecoderModel(Model):
    def __init__(self, device: Device, model_dir: str, base_size: list[int] | None = None) -> None:
        super().__init__(
            id="inspyre-net:decoder",
            gpu_mem=GPU_MEM,
            cpu_mem=CPU_MEM,
            device=device,
        )
        self._model_dir = model_dir
        self._base_size = base_size or [1024, 1024]
        self._decoder: _Decoder | None = None
        self._image_pyramid: ImagePyramid | None = None

    def load(self) -> None:
        if self._decoder is not None:
            return

        ckpt_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=CHECKPOINT_FILENAME,
            cache_dir=self._model_dir,
        )
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        decoder_state = {
            k: v for k, v in state_dict.items()
            if any(k.startswith(p) for p in DECODER_PREFIXES)
        }

        decoder = _Decoder(IN_CHANNELS, DEPTH, self._base_size)
        missing, unexpected = decoder.load_state_dict(decoder_state, strict=False)
        if unexpected:
            raise RuntimeError(
                f"InSPyReNet decoder got unexpected keys: {unexpected[:3]}...",
            )
        decoder.to(device=DEVICE, dtype=DTYPE)
        decoder.eval()
        self._decoder = decoder

        self._image_pyramid = ImagePyramid(7, 1).to(device=DEVICE, dtype=DTYPE)

    def unload(self) -> None:
        if self._decoder is None:
            return
        del self._decoder
        self._decoder = None
        self._image_pyramid = None
        torch.cuda.empty_cache()

    @torch.no_grad()
    def pipe(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        *,
        features: tuple[torch.Tensor, ...],
        target_hw: tuple[int, int],
    ) -> dict[str, list[torch.Tensor]]:
        assert self._decoder is not None
        assert self._image_pyramid is not None

        H, W = target_hw
        x1, x2, x3, x4, x5 = features

        x1 = self._decoder.context1(x1)
        x2 = self._decoder.context2(x2)
        x3 = self._decoder.context3(x3)
        x4 = self._decoder.context4(x4)
        x5 = self._decoder.context5(x5)

        f3, d3 = self._decoder.decoder([x3, x4, x5])

        f3 = _resize(f3, (H // 4, W // 4))
        f2, p2 = self._decoder.attention2(torch.cat([x2, f3], dim=1), d3.detach())
        d2 = self._image_pyramid.reconstruct(d3.detach(), p2)

        x1 = _resize(x1, (H // 2, W // 2))
        f2 = _resize(f2, (H // 2, W // 2))
        f1, p1 = self._decoder.attention1(torch.cat([x1, f2], dim=1), d2.detach(), p2.detach())
        d1 = self._image_pyramid.reconstruct(d2.detach(), p1)

        f1 = _resize(f1, (H, W))
        _, p0 = self._decoder.attention0(f1, d1.detach(), p1.detach())
        d0 = self._image_pyramid.reconstruct(d1.detach(), p0)

        return {
            "saliency": [d3, d2, d1, d0],
            "laplacian": [p2, p1, p0],
        }