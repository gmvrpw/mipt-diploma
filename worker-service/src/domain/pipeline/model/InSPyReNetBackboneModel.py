import torch
from huggingface_hub import hf_hub_download

from src.domain.pipeline.model.Model import Model
from src.domain.pipeline.model.inspyre_net import SwinB
from src.domain.pipeline.model.inspyre_net.swin import SwinTransformer
from src.domain.worker.device.Device import Device


GB = 1024 ** 3
MB = 1024 ** 2
WEIGHTS_GPU_MEM = 512 * MB
ACTIVATIONS_GPU_MEM = 3584 * MB
GPU_MEM = WEIGHTS_GPU_MEM + ACTIVATIONS_GPU_MEM
CPU_MEM = 1 * GB

REPO_ID = "plemeri/InSPyReNet"
CHECKPOINT_FILENAME = "snapshots/Plus_Ultra/latest.pth"
PREFIX = "backbone."

DEVICE = "cuda"
DTYPE = torch.float32


class InSPyReNetBackboneModel(Model):
    def __init__(self, device: Device, model_dir: str) -> None:
        super().__init__(
            id="inspyre-net:swinb-backbone",
            gpu_mem=GPU_MEM,
            cpu_mem=CPU_MEM,
            device=device,
        )
        self._model_dir = model_dir
        self._backbone: SwinTransformer | None = None

    def load(self) -> None:
        if self._backbone is not None:
            return

        ckpt_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=CHECKPOINT_FILENAME,
            cache_dir=self._model_dir,
        )
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        backbone_state = {
            k[len(PREFIX):]: v
            for k, v in state_dict.items()
            if k.startswith(PREFIX)
        }

        backbone = SwinB()
        missing, unexpected = backbone.load_state_dict(backbone_state, strict=False)
        if unexpected:
            raise RuntimeError(
                f"InSPyReNet backbone got unexpected keys from checkpoint: {unexpected[:3]}...",
            )
        backbone.to(device=DEVICE, dtype=DTYPE)
        backbone.eval()
        self._backbone = backbone

    def unload(self) -> None:
        if self._backbone is None:
            return
        del self._backbone
        self._backbone = None
        torch.cuda.empty_cache()

    @torch.no_grad()
    def pipe(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        *,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        assert self._backbone is not None
        x = x.to(device=DEVICE, dtype=DTYPE)
        return self._backbone(x)