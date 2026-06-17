import torch
from diffusers.models.transformers.transformer_wan import WanTransformer3DModel

from src.domain.pipeline.model.Model import Model
from src.domain.worker.device.Device import Device


GB = 1024 ** 3
WEIGHTS_GPU_MEM = 28 * GB
ACTIVATIONS_GPU_MEM = 6 * GB
GPU_MEM = WEIGHTS_GPU_MEM + ACTIVATIONS_GPU_MEM
CPU_MEM = 28 * GB

DEVICE = "cuda"
DTYPE = torch.bfloat16


class WanTransformerModel(Model):
    def __init__(
        self, device: Device, model_dir: str, hf_path: str, subfolder: str,
    ) -> None:
        super().__init__(
            id=f"{hf_path}:{subfolder}",
            gpu_mem=GPU_MEM,
            cpu_mem=CPU_MEM,
            device=device,
        )
        self._model_dir = model_dir
        self._hf_path = hf_path
        self._subfolder = subfolder
        self._transformer: WanTransformer3DModel | None = None

        config = WanTransformer3DModel.load_config(
            hf_path, cache_dir=model_dir, subfolder=subfolder,
        )
        self._patch_size: tuple[int, ...] = tuple(config["patch_size"])

    def load(self) -> None:
        if self._transformer is not None:
            return

        transformer = WanTransformer3DModel.from_pretrained(
            self._hf_path,
            cache_dir=self._model_dir,
            subfolder=self._subfolder,
            torch_dtype=DTYPE,
        )
        transformer.to(DEVICE)  # type: ignore[arg-type]
        transformer.eval()
        self._transformer = transformer

    def unload(self) -> None:
        if self._transformer is None:
            return

        del self._transformer
        self._transformer = None
        torch.cuda.empty_cache()

    @property
    def dtype(self) -> torch.dtype:
        assert self._transformer is not None

        return self._transformer.dtype

    @property
    def patch_size(self) -> tuple[int, ...]:
        return self._patch_size

    def cache_context(self, name: str):
        assert self._transformer is not None

        return self._transformer.cache_context(name)

    @torch.no_grad()
    def pipe(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        *,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_kwargs: dict | None = None,
    ) -> torch.Tensor:
        assert self._transformer is not None

        return self._transformer(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            attention_kwargs=attention_kwargs or {},
            return_dict=False,
        )[0]
