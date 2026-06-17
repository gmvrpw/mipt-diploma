import torch
from diffusers.models.autoencoders.autoencoder_kl_wan import AutoencoderKLWan

from src.domain.pipeline.model.Model import Model
from src.domain.worker.device.Device import Device
from src.domain.worker.device.eviction.event import ModelInferencedEvent


GB = 1024 ** 3
WEIGHTS_GPU_MEM = 1 * GB
ACTIVATIONS_GPU_MEM = 5 * GB
GPU_MEM = WEIGHTS_GPU_MEM + ACTIVATIONS_GPU_MEM
CPU_MEM = 3 * GB

SUBFOLDER = "vae"

DEVICE = "cuda"
DTYPE = torch.float32


class WanVaeModel(Model):
    def __init__(self, device: Device, model_dir: str, hf_path: str) -> None:
        super().__init__(
            id=f"{hf_path}:{SUBFOLDER}",
            gpu_mem=GPU_MEM,
            cpu_mem=CPU_MEM,
            device=device,
        )
        self._model_dir = model_dir
        self._hf_path = hf_path
        self._vae: AutoencoderKLWan | None = None
        self._latents_mean: torch.Tensor | None = None
        self._latents_inv_std: torch.Tensor | None = None

        config = AutoencoderKLWan.load_config(
            hf_path, cache_dir=model_dir, subfolder=SUBFOLDER,
        )
        self._z_dim = int(config["z_dim"])

        downsample = config.get("temperal_downsample", [False, True, True])
        self._scale_factor_spatial = int(
            config.get("scale_factor_spatial", 2 ** len(downsample)),
        )
        self._scale_factor_temporal = int(
            config.get("scale_factor_temporal", 2 ** sum(downsample)),
        )

    def load(self) -> None:
        if self._vae is not None:
            return

        self._vae = AutoencoderKLWan.from_pretrained(
            self._hf_path,
            cache_dir=self._model_dir,
            subfolder=SUBFOLDER,
            torch_dtype=DTYPE,
        )
        self._vae.to(DEVICE)  # type: ignore[arg-type]
        self._vae.eval()

        z_dim = int(getattr(self._vae.config, "z_dim"))
        latents_mean = getattr(self._vae.config, "latents_mean")
        latents_std = getattr(self._vae.config, "latents_std")

        self._latents_mean = torch.tensor(latents_mean).view(1, z_dim, 1, 1, 1)
        self._latents_inv_std = 1.0 / torch.tensor(latents_std).view(1, z_dim, 1, 1, 1)

    def unload(self) -> None:
        if self._vae is None:
            return

        del self._vae
        self._vae = None
        self._latents_mean = None
        self._latents_inv_std = None
        torch.cuda.empty_cache()

    @property
    def dtype(self) -> torch.dtype:
        assert self._vae is not None

        return self._vae.dtype

    @property
    def z_dim(self) -> int:
        return self._z_dim

    @property
    def scale_factor_spatial(self) -> int:
        return self._scale_factor_spatial

    @property
    def scale_factor_temporal(self) -> int:
        return self._scale_factor_temporal

    @torch.no_grad()
    def pipe_encode(self, video: torch.Tensor) -> torch.Tensor:
        self._device.load(self)
        assert self._vae is not None
        assert self._latents_mean is not None
        assert self._latents_inv_std is not None

        video = video.to(device=DEVICE, dtype=DTYPE)
        (latent_dist,) = self._vae.encode(video, return_dict=False)
        latents = latent_dist.mode()
        mean = self._latents_mean.to(latents.device, latents.dtype)
        inv_std = self._latents_inv_std.to(latents.device, latents.dtype)
        out = (latents - mean) * inv_std
        self._device.signal(ModelInferencedEvent(self.id))
        return out

    @torch.no_grad()
    def pipe_decode(self, latents: torch.Tensor) -> torch.Tensor:
        self._device.load(self)
        assert self._vae is not None
        assert self._latents_mean is not None
        assert self._latents_inv_std is not None

        latents = latents.to(device=DEVICE, dtype=DTYPE)
        mean = self._latents_mean.to(latents.device, latents.dtype)
        inv_std = self._latents_inv_std.to(latents.device, latents.dtype)
        latents = latents / inv_std + mean
        out = self._vae.decode(latents, return_dict=False)[0]
        self._device.signal(ModelInferencedEvent(self.id))
        return out

    def pipe(self, **_kwargs):
        raise NotImplementedError("Use pipe_encode or pipe_decode")
