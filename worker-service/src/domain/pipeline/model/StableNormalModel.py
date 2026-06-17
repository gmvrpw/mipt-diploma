import torch
from PIL.Image import Image

from src.domain.pipeline.model.Model import Model
from src.domain.pipeline.model.stable_normal import (
    HEURI_DDIMScheduler,
    StableNormalPipeline,
    YOSONormalsPipeline,
)
from src.domain.worker.device.Device import Device


GB = 1024 ** 3
WEIGHTS_GPU_MEM = 5 * GB
ACTIVATIONS_GPU_MEM = 3 * GB
GPU_MEM = WEIGHTS_GPU_MEM + ACTIVATIONS_GPU_MEM
CPU_MEM = 8 * GB

YOSO_REPO_ID = "Stable-X/yoso-normal-v0-3"
MAIN_REPO_ID = "Stable-X/stable-normal-v0-1"

DEVICE = "cuda"
DTYPE = torch.float16


class StableNormalModel(Model):
    def __init__(self, device: Device, model_dir: str) -> None:
        super().__init__(
            id="stable-normal-v0-1",
            gpu_mem=GPU_MEM,
            cpu_mem=CPU_MEM,
            device=device,
        )
        self._model_dir = model_dir
        self._pipe: StableNormalPipeline | None = None

    def load(self) -> None:
        if self._pipe is not None:
            return

        x_start_pipeline = YOSONormalsPipeline.from_pretrained(
            YOSO_REPO_ID,
            cache_dir=self._model_dir,
            variant="fp16",
            torch_dtype=DTYPE,
            trust_remote_code=True,
            safety_checker=None,
        )
        assert isinstance(x_start_pipeline, YOSONormalsPipeline)
        x_start_pipeline.to(DEVICE)

        scheduler = HEURI_DDIMScheduler(
            prediction_type="sample",
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
        )
        pipe = StableNormalPipeline.from_pretrained(
            MAIN_REPO_ID,
            cache_dir=self._model_dir,
            variant="fp16",
            torch_dtype=DTYPE,
            trust_remote_code=True,
            scheduler=scheduler,
            safety_checker=None,
        )
        assert isinstance(pipe, StableNormalPipeline)
        pipe.x_start_pipeline = x_start_pipeline
        pipe.to(DEVICE)
        pipe.prior.to(DEVICE, DTYPE)

        self._pipe = pipe

    def unload(self) -> None:
        if self._pipe is None:
            return
        self._pipe.x_start_pipeline = None
        self._pipe.prior = None
        del self._pipe
        self._pipe = None
        torch.cuda.empty_cache()

    @torch.no_grad()
    def pipe(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        *,
        image: Image | list[Image] | torch.Tensor,
        num_inference_steps: int = 10,
        match_input_resolution: bool = False,
    ) -> torch.Tensor:
        assert self._pipe is not None

        out = self._pipe(
            image,
            num_inference_steps=num_inference_steps,
            match_input_resolution=match_input_resolution,
            output_type="pt",
        )
        prediction = out.prediction
        if not isinstance(prediction, torch.Tensor):
            prediction = torch.from_numpy(prediction)
        return prediction