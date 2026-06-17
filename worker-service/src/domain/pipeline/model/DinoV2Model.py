import torch
from transformers import Dinov2Model as HfDinov2Model

from src.domain.pipeline.model.Model import Model
from src.domain.worker.device.Device import Device


GB = 1024 ** 3
MB = 1024 ** 2
WEIGHTS_GPU_MEM = 128 * MB
ACTIVATIONS_GPU_MEM = 896 * MB
GPU_MEM = WEIGHTS_GPU_MEM + ACTIVATIONS_GPU_MEM
CPU_MEM = 1 * GB

CHECKPOINT = "facebook/dinov2-small"

DEVICE = "cuda"
DTYPE = torch.float32


class DinoV2Model(Model):
    def __init__(self, device: Device, model_dir: str) -> None:
        super().__init__(
            id=CHECKPOINT,
            gpu_mem=GPU_MEM,
            cpu_mem=CPU_MEM,
            device=device,
        )
        self._model_dir = model_dir
        self._model: HfDinov2Model | None = None

    def load(self) -> None:
        if self._model is not None:
            return

        model = HfDinov2Model.from_pretrained(
            CHECKPOINT,
            cache_dir=self._model_dir,
            torch_dtype=DTYPE,
        )
        model.to(DEVICE)  # type: ignore[arg-type]
        model.eval()
        self._model = model

    def unload(self) -> None:
        if self._model is None:
            return

        del self._model
        self._model = None
        torch.cuda.empty_cache()

    @torch.no_grad()
    def pipe(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        *,
        pixel_values: torch.Tensor,
    ) -> torch.Tensor:
        assert self._model is not None

        pixel_values = pixel_values.to(device=DEVICE, dtype=DTYPE)
        out = self._model(pixel_values=pixel_values)
        return out.last_hidden_state[:, 0, :]
