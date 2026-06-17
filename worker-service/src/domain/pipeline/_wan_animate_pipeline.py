from __future__ import annotations

import html
from collections.abc import Callable
from typing import Any, ClassVar, Literal, cast

import ftfy
import regex as re
import structlog
import torch
from diffusers.image_processor import PipelineImageInput
from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from PIL import Image
from structlog.stdlib import BoundLogger
from transformers import AutoTokenizer

from src.domain.pipeline.model.WanTextEncoderModel import WanTextEncoderModel
from src.domain.pipeline.model.WanTransformerModel import WanTransformerModel
from src.domain.pipeline.model.WanVaeModel import WanVaeModel
from src.domain.worker.device.Device import Device
from src.domain.worker.device.eviction.event import ModelInferencedEvent


log: BoundLogger = structlog.get_logger(__name__)

DEVICE = "cuda"

TRANSFORMER_SUBFOLDER = "transformer"
TRANSFORMER_2_SUBFOLDER = "transformer_2"

DEFAULT_HEIGHT = 480
DEFAULT_WIDTH = 832
DEFAULT_NUM_FRAMES = 81
DEFAULT_NUM_INFERENCE_STEPS = 40
DEFAULT_GUIDANCE_SCALE_HIGH = 3.5
DEFAULT_GUIDANCE_SCALE_LOW = 3.5
DEFAULT_MAX_SEQUENCE_LENGTH = 512
DEFAULT_BOUNDARY_RATIO = 0.875

NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，"
    "静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，"
    "多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，"
    "形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，"
    "背景人很多，倒着走"
)


OnStep = Callable[[], None]


def _fire(on_step: OnStep | None) -> None:
    if on_step is not None:
        on_step()


def _basic_clean(text: str) -> str:
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def _whitespace_clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _prompt_clean(text: str) -> str:
    return _whitespace_clean(_basic_clean(text))


def _round_num_frames(num_frames: int, temporal_factor: int) -> int:
    if num_frames % temporal_factor != 1:
        rounded = num_frames // temporal_factor * temporal_factor + 1
        log.warning(
            "num_frames adjusted to satisfy temporal divisibility",
            requested=num_frames, rounded=rounded, temporal_factor=temporal_factor,
        )
        num_frames = rounded
    return max(num_frames, 1)


def _round_dims(
    height: int, width: int, vae_spatial: int, patch_size: tuple[int, ...],
) -> tuple[int, int]:
    h_multiple = vae_spatial * patch_size[1]
    w_multiple = vae_spatial * patch_size[2]
    calc_h = height // h_multiple * h_multiple
    calc_w = width // w_multiple * w_multiple
    if calc_h != height or calc_w != width:
        log.warning(
            "height/width adjusted to satisfy patchification",
            requested=(height, width), rounded=(calc_h, calc_w),
        )
    return calc_h, calc_w


class Wan22ImageToVideoPipeline:
    _shared: ClassVar[dict[tuple[int, str, str],
                           Wan22ImageToVideoPipeline]] = {}

    @classmethod
    def shared(
        cls, device: Device, model_dir: str, hf_path: str,
    ) -> Wan22ImageToVideoPipeline:
        key = (id(device), model_dir, hf_path)
        if key not in cls._shared:
            cls._shared[key] = cls(
                device=device, model_dir=model_dir, hf_path=hf_path,
            )
        return cls._shared[key]

    def __init__(
        self, device: Device, model_dir: str, hf_path: str,
    ) -> None:
        self._device = device
        self._vae = WanVaeModel(
            device=device, model_dir=model_dir, hf_path=hf_path,
        )
        self._text_encoder = WanTextEncoderModel(
            device=device, model_dir=model_dir, hf_path=hf_path,
        )
        self._transformer_high = WanTransformerModel(
            device=device, model_dir=model_dir, hf_path=hf_path,
            subfolder=TRANSFORMER_SUBFOLDER,
        )
        self._transformer_low = WanTransformerModel(
            device=device, model_dir=model_dir, hf_path=hf_path,
            subfolder=TRANSFORMER_2_SUBFOLDER,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            hf_path, cache_dir=model_dir, subfolder="tokenizer",
        )
        self._tokenizer = tokenizer

        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            hf_path, cache_dir=model_dir, subfolder="scheduler",
        )
        assert isinstance(scheduler, FlowMatchEulerDiscreteScheduler)
        self._scheduler = scheduler

        self._video_processor = VideoProcessor(vae_scale_factor=8)

    @torch.no_grad()
    def __call__(
        self,
        image: Image.Image,
        prompt: str,
        *,
        last_image: Image.Image | None = None,
        negative_prompt: str | None = None,
        height: int = DEFAULT_HEIGHT,
        width: int = DEFAULT_WIDTH,
        num_frames: int = DEFAULT_NUM_FRAMES,
        num_inference_steps: int = DEFAULT_NUM_INFERENCE_STEPS,
        guidance_scale: float = DEFAULT_GUIDANCE_SCALE_HIGH,
        guidance_scale_2: float = DEFAULT_GUIDANCE_SCALE_LOW,
        boundary_ratio: float = DEFAULT_BOUNDARY_RATIO,
        max_sequence_length: int = DEFAULT_MAX_SEQUENCE_LENGTH,
        generator: torch.Generator | None = None,
        latents: torch.Tensor | None = None,
        attention_kwargs: dict[str, Any] | None = None,
        output_type: Literal["pil", "latent"] = "pil",
        on_step: OnStep | None = None,
    ) -> list[list[Image.Image]] | torch.Tensor:
        if negative_prompt is None:
            negative_prompt = NEGATIVE_PROMPT

        patch_size = self._transformer_high.patch_size
        vae_spatial = self._vae.scale_factor_spatial
        vae_temporal = self._vae.scale_factor_temporal

        height, width = _round_dims(height, width, vae_spatial, patch_size)
        num_frames = _round_num_frames(num_frames, vae_temporal)

        do_cfg = guidance_scale > 1 or guidance_scale_2 > 1

        with self._text_encoder:
            prompt_embeds = self._encode_prompt(
                prompt, max_sequence_length,
            )
            self._device.signal(ModelInferencedEvent(self._text_encoder.id))
            _fire(on_step)

            negative_prompt_embeds = self._encode_prompt(
                negative_prompt, max_sequence_length,
            ) if do_cfg else None
            if negative_prompt_embeds is not None:
                self._device.signal(
                    ModelInferencedEvent(self._text_encoder.id))
                _fire(on_step)

        transformer_dtype = torch.bfloat16
        prompt_embeds = prompt_embeds.to(transformer_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(
                transformer_dtype)

        image_tensor = self._video_processor.preprocess(
            cast(PipelineImageInput, image), height=height, width=width,
        ).to(DEVICE, dtype=torch.float32)
        last_image_tensor: torch.Tensor | None = None
        if last_image is not None:
            last_image_tensor = self._video_processor.preprocess(
                cast(PipelineImageInput, last_image), height=height, width=width,
            ).to(DEVICE, dtype=torch.float32)

        with self._vae:
            latents, condition = self._prepare_latents(
                image_tensor, last_image_tensor,
                num_channels_latents=self._vae.z_dim,
                height=height, width=width, num_frames=num_frames,
                vae_spatial=vae_spatial, vae_temporal=vae_temporal,
                dtype=torch.float32, generator=generator, latents=latents,
            )
            self._device.signal(ModelInferencedEvent(self._vae.id))
        _fire(on_step)

        self._scheduler.set_timesteps(num_inference_steps, device=DEVICE)
        timesteps = cast(torch.Tensor, self._scheduler.timesteps)
        num_train_timesteps = self._scheduler.config.get(
            "num_train_timesteps", 1000)
        boundary_timestep = boundary_ratio * num_train_timesteps

        latents = self._denoise(
            latents=latents,
            condition=condition,
            timesteps=timesteps,
            boundary_timestep=boundary_timestep,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            guidance_scale=guidance_scale,
            guidance_scale_2=guidance_scale_2,
            transformer_dtype=transformer_dtype,
            do_cfg=do_cfg,
            attention_kwargs=attention_kwargs,
            on_step=on_step,
        )

        if output_type == "latent":
            return latents

        with self._vae:
            decoded = self._vae.pipe_decode(latents)
            self._device.signal(ModelInferencedEvent(self._vae.id))
        _fire(on_step)

        video = self._video_processor.postprocess_video(
            decoded, output_type="pil",
        )
        return cast(list[list[Image.Image]], video)

    def _encode_prompt(
        self, prompt: str, max_sequence_length: int,
    ) -> torch.Tensor:
        cleaned = _prompt_clean(prompt)
        text_inputs = self._tokenizer(
            [cleaned],
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids = text_inputs.input_ids.to(DEVICE)
        mask = text_inputs.attention_mask.to(DEVICE)
        seq_lens = mask.gt(0).sum(dim=1).long()

        prompt_embeds = self._text_encoder.pipe(
            input_ids=input_ids, attention_mask=mask,
        )
        prompt_embeds = prompt_embeds.to(
            dtype=self._text_encoder.dtype, device=DEVICE,
        )
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
        prompt_embeds = torch.stack([
            torch.cat(
                [u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))])
            for u in prompt_embeds
        ], dim=0)
        return prompt_embeds

    def _prepare_latents(
        self,
        image: torch.Tensor,
        last_image: torch.Tensor | None,
        *,
        num_channels_latents: int,
        height: int,
        width: int,
        num_frames: int,
        vae_spatial: int,
        vae_temporal: int,
        dtype: torch.dtype,
        generator: torch.Generator | None,
        latents: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_latent_frames = (num_frames - 1) // vae_temporal + 1
        latent_height = height // vae_spatial
        latent_width = width // vae_spatial
        batch_size = 1

        shape = (
            batch_size, num_channels_latents,
            num_latent_frames, latent_height, latent_width,
        )

        if latents is None:
            latents = randn_tensor(
                shape, generator=generator,
                device=torch.device(DEVICE), dtype=dtype,
            )
        else:
            latents = latents.to(device=DEVICE, dtype=dtype)

        image = image.unsqueeze(2)  # [B, C, 1, H, W]
        if last_image is None:
            video_condition = torch.cat(
                [
                    image,
                    image.new_zeros(
                        image.shape[0], image.shape[1],
                        num_frames - 1, height, width,
                    ),
                ],
                dim=2,
            )
        else:
            last_image = last_image.unsqueeze(2)
            video_condition = torch.cat(
                [
                    image,
                    image.new_zeros(
                        image.shape[0], image.shape[1],
                        num_frames - 2, height, width,
                    ),
                    last_image,
                ],
                dim=2,
            )
        video_condition = video_condition.to(
            device=DEVICE, dtype=self._vae.dtype)

        latent_condition = self._vae.pipe_encode(video_condition)
        latent_condition = latent_condition.repeat(batch_size, 1, 1, 1, 1)
        latent_condition = latent_condition.to(dtype)

        mask_lat_size = torch.ones(
            batch_size, 1, num_frames, latent_height, latent_width,
        )
        if last_image is None:
            mask_lat_size[:, :, list(range(1, num_frames))] = 0
        else:
            mask_lat_size[:, :, list(range(1, num_frames - 1))] = 0
        first_frame_mask = mask_lat_size[:, :, 0:1]
        first_frame_mask = torch.repeat_interleave(
            first_frame_mask, dim=2, repeats=vae_temporal,
        )
        mask_lat_size = torch.cat(
            [first_frame_mask, mask_lat_size[:, :, 1:, :]], dim=2,
        )
        mask_lat_size = mask_lat_size.view(
            batch_size, -1, vae_temporal, latent_height, latent_width,
        )
        mask_lat_size = mask_lat_size.transpose(
            1, 2).to(latent_condition.device)

        condition = torch.cat([mask_lat_size, latent_condition], dim=1)
        return latents, condition

    def _denoise(
        self,
        *,
        latents: torch.Tensor,
        condition: torch.Tensor,
        timesteps: torch.Tensor,
        boundary_timestep: float,
        prompt_embeds: torch.Tensor,
        negative_prompt_embeds: torch.Tensor | None,
        guidance_scale: float,
        guidance_scale_2: float,
        transformer_dtype: torch.dtype,
        do_cfg: bool,
        attention_kwargs: dict[str, Any] | None,
        on_step: OnStep | None,
    ) -> torch.Tensor:
        high_idx = [i for i, t in enumerate(
            timesteps) if t >= boundary_timestep]
        low_idx = [i for i, t in enumerate(timesteps) if t < boundary_timestep]

        if high_idx:
            with self._transformer_high:
                latents = self._denoise_segment(
                    transformer=self._transformer_high,
                    latents=latents,
                    condition=condition,
                    timesteps=timesteps[high_idx[0]:high_idx[-1] + 1],
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=negative_prompt_embeds,
                    current_guidance_scale=guidance_scale,
                    transformer_dtype=transformer_dtype,
                    do_cfg=do_cfg,
                    attention_kwargs=attention_kwargs,
                    on_step=on_step,
                )

        if low_idx:
            with self._transformer_low:
                latents = self._denoise_segment(
                    transformer=self._transformer_low,
                    latents=latents,
                    condition=condition,
                    timesteps=timesteps[low_idx[0]:low_idx[-1] + 1],
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=negative_prompt_embeds,
                    current_guidance_scale=guidance_scale_2,
                    transformer_dtype=transformer_dtype,
                    do_cfg=do_cfg,
                    attention_kwargs=attention_kwargs,
                    on_step=on_step,
                )

        return latents

    def _denoise_segment(
        self,
        *,
        transformer: WanTransformerModel,
        latents: torch.Tensor,
        condition: torch.Tensor,
        timesteps: torch.Tensor,
        prompt_embeds: torch.Tensor,
        negative_prompt_embeds: torch.Tensor | None,
        current_guidance_scale: float,
        transformer_dtype: torch.dtype,
        do_cfg: bool,
        attention_kwargs: dict[str, Any] | None,
        on_step: OnStep | None,
    ) -> torch.Tensor:
        for t in timesteps:
            latent_model_input = torch.cat(
                [latents, condition], dim=1,
            ).to(transformer_dtype)
            timestep = t.expand(latents.shape[0])

            with transformer.cache_context("cond"):
                noise_pred = transformer.pipe(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    attention_kwargs=attention_kwargs,
                )
            self._device.signal(ModelInferencedEvent(transformer.id))

            if do_cfg:
                assert negative_prompt_embeds is not None
                with transformer.cache_context("uncond"):
                    noise_uncond = transformer.pipe(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=negative_prompt_embeds,
                        attention_kwargs=attention_kwargs,
                    )
                self._device.signal(ModelInferencedEvent(transformer.id))
                noise_pred = noise_uncond + current_guidance_scale * (
                    noise_pred - noise_uncond
                )

            stepped = cast(Any, self._scheduler).step(
                noise_pred, t, latents, return_dict=False,
            )
            latents = cast(torch.Tensor, stepped[0])

            _fire(on_step)

        return latents
