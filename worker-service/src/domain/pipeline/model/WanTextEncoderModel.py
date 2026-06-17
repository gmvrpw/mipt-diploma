import torch
from transformers import UMT5EncoderModel

from src.domain.pipeline.model.Model import Model
from src.domain.worker.device.Device import Device


GB = 1024 ** 3
WEIGHTS_GPU_MEM = 11 * GB
ACTIVATIONS_GPU_MEM = 3 * GB
GPU_MEM = WEIGHTS_GPU_MEM + ACTIVATIONS_GPU_MEM
CPU_MEM = 14 * GB

SUBFOLDER = "text_encoder"

DEVICE = "cuda"
DTYPE = torch.bfloat16


class WanTextEncoderModel(Model):
    def __init__(self, device: Device, model_dir: str, hf_path: str) -> None:
        super().__init__(
            id=f"{hf_path}:{SUBFOLDER}",
            gpu_mem=GPU_MEM,
            cpu_mem=CPU_MEM,
            device=device,
        )
        self._model_dir = model_dir
        self._hf_path = hf_path
        self._encoder: UMT5EncoderModel | None = None

    def load(self) -> None:
        if self._encoder is not None:
            return

        self._encoder = UMT5EncoderModel.from_pretrained(
            self._hf_path,
            cache_dir=self._model_dir,
            subfolder=SUBFOLDER,
            torch_dtype=DTYPE,
        )
        self._encoder.to(DEVICE)  # type: ignore[arg-type]
        self._encoder.eval()

    def unload(self) -> None:
        if self._encoder is None:
            return

        del self._encoder
        self._encoder = None
        torch.cuda.empty_cache()

    @property
    def dtype(self) -> torch.dtype:
        assert self._encoder is not None

        return self._encoder.dtype

    @torch.no_grad()
    def pipe(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        assert self._encoder is not None

        outputs = self._encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return outputs.last_hidden_state
