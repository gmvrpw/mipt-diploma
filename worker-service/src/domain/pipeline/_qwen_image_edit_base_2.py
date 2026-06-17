from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, ClassVar, Literal, cast

import numpy as np
import torch
from diffusers.image_processor import VaeImageProcessor
from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image
from transformers import Qwen2VLProcessor

from src.domain.pipeline.model.QwenTextEncoderModel import QwenTextEncoderModel
from src.domain.pipeline.model.QwenTransformerModel import QwenTransformerModel
from src.domain.pipeline.model.QwenVaeModel import QwenVaeModel
from src.domain.worker.device.Device import Device
from src.domain.worker.device.eviction.event import ModelInferencedEvent


DEVICE = "cuda"

VAE_SCALE_FACTOR = 8
PATCH_SIZE = 2
PACK_FACTOR = VAE_SCALE_FACTOR * PATCH_SIZE

NUM_CHANNELS_LATENTS = 64 // 4
DIMENSION_GRID = 32

CONDITION_IMAGE_AREA = 384 * 384
VAE_IMAGE_AREA = 1024 * 1024

DEFAULT_TRUE_CFG_SCALE = 4.0
DEFAULT_NUM_INFERENCE_STEPS = 50
DEFAULT_MAX_SEQUENCE_LENGTH = 512

PROMPT_TEMPLATE = (
    "<|im_start|>system\nDescribe the key features of the input image (color, "
    "shape, size, texture, objects, background), then explain how the user's "
    "text instruction should alter or modify the image. Generate a new image "
    "that meets the user's requirements while maintaining consistency with "
    "the original input where appropriate.<|im_end|>\n"
    "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
)
PROMPT_TEMPLATE_START_IDX = 64
IMG_PROMPT_TEMPLATE = "Picture {}: <|vision_start|><|image_pad|><|vision_end|>"


OnStep = Callable[[], None]


def _calculate_dimensions(target_area: int, ratio: float) -> tuple[int, int]:
    width = math.sqrt(target_area * ratio)
    height = width / ratio
    width = round(width / DIMENSION_GRID) * DIMENSION_GRID
    height = round(height / DIMENSION_GRID) * DIMENSION_GRID
    return width, height


def _calculate_shift(
    image_seq_len: int,
    base_seq_len: int,
    max_seq_len: int,
    base_shift: float,
    max_shift: float,
) -> float:
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


def _extract_masked_hidden(
    hidden_states: torch.Tensor, mask: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    bool_mask = mask.bool()
    valid_lengths = bool_mask.sum(dim=1)
    selected = hidden_states[bool_mask]
    return torch.split(selected, valid_lengths.tolist(), dim=0)


def _pack_latents(
    latents: torch.Tensor,
    batch_size: int,
    num_channels: int,
    height: int,
    width: int,
) -> torch.Tensor:
    latents = latents.view(
        batch_size, num_channels, height // 2, 2, width // 2, 2,
    )
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    return latents.reshape(
        batch_size, (height // 2) * (width // 2), num_channels * 4,
    )


def _unpack_latents(latents: torch.Tensor, height: int, width: int) -> torch.Tensor:
    batch_size, _, channels = latents.shape
    h = 2 * (int(height) // PACK_FACTOR)
    w = 2 * (int(width) // PACK_FACTOR)
    latents = latents.view(batch_size, h // 2, w // 2, channels // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)
    return latents.reshape(batch_size, channels // 4, 1, h, w)


def _fire(on_step: OnStep | None) -> None:
    if on_step is not None:
        on_step()


class QwenImageEditPlusPipeline:
    _shared: ClassVar[dict[tuple[int, str, str],
                           QwenImageEditPlusPipeline]] = {}

    @classmethod
    def shared(
        cls, device: Device, model_dir: str, hf_path: str,
    ) -> QwenImageEditPlusPipeline:
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
        self._vae = QwenVaeModel(
            device=device, model_dir=model_dir, hf_path=hf_path,
        )
        self._text_encoder = QwenTextEncoderModel(
            device=device, model_dir=model_dir, hf_path=hf_path,
        )
        self._transformer = QwenTransformerModel(
            device=device, model_dir=model_dir, hf_path=hf_path,
        )

        processor = Qwen2VLProcessor.from_pretrained(
            hf_path, cache_dir=model_dir, subfolder="processor",
        )
        assert isinstance(processor, Qwen2VLProcessor)
        self._processor = processor

        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            hf_path, cache_dir=model_dir, subfolder="scheduler",
        )
        assert isinstance(scheduler, FlowMatchEulerDiscreteScheduler)
        self._scheduler = scheduler

        self._image_processor = VaeImageProcessor(vae_scale_factor=PACK_FACTOR)

    @torch.no_grad()
    def __call__(
        self,
        image: Image.Image | list[Image.Image],
        prompt: str | None = None,
        *,
        negative_prompt: str | None = None,
        true_cfg_scale: float = DEFAULT_TRUE_CFG_SCALE,
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int = DEFAULT_NUM_INFERENCE_STEPS,
        sigmas: list[float] | None = None,
        guidance_scale: float | None = None,
        generator: torch.Generator | None = None,
        latents: torch.Tensor | None = None,
        prompt_embeds: torch.Tensor | None = None,
        prompt_embeds_mask: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds_mask: torch.Tensor | None = None,
        attention_kwargs: dict[str, Any] | None = None,
        max_sequence_length: int = DEFAULT_MAX_SEQUENCE_LENGTH,
        output_type: Literal["pil", "latent"] = "pil",
        on_step: OnStep | None = None,
    ) -> list[Image.Image] | torch.Tensor:
        self._validate_inputs(
            prompt, prompt_embeds, negative_prompt, negative_prompt_embeds,
            max_sequence_length,
        )

        images = [image] if isinstance(image, Image.Image) else list(image)
        images = [img.convert("RGB") for img in images]

        ref_w, ref_h = images[-1].size
        calc_w, calc_h = _calculate_dimensions(VAE_IMAGE_AREA, ref_w / ref_h)
        height = (height or calc_h) // PACK_FACTOR * PACK_FACTOR
        width = (width or calc_w) // PACK_FACTOR * PACK_FACTOR

        condition_images, vae_images, vae_image_sizes = self._preprocess(
            images)
        _fire(on_step)

        do_true_cfg = true_cfg_scale > 1 and (
            negative_prompt is not None or negative_prompt_embeds is not None
        )

        with self._text_encoder:
            if prompt_embeds is None:
                assert prompt is not None
                prompt_embeds, prompt_embeds_mask = self._encode_prompt(
                    prompt, condition_images,
                )
                self._device.signal(
                    ModelInferencedEvent(self._text_encoder.id))
                _fire(on_step)

            if do_true_cfg and negative_prompt_embeds is None:
                neg_text = negative_prompt if negative_prompt is not None else " "
                negative_prompt_embeds, negative_prompt_embeds_mask = self._encode_prompt(
                    neg_text, condition_images,
                )
                self._device.signal(
                    ModelInferencedEvent(self._text_encoder.id))
                _fire(on_step)

        with self._vae:
            image_latents_5d = [self._vae.pipe_encode(t) for t in vae_images]
            self._device.signal(ModelInferencedEvent(self._vae.id))
        _fire(on_step)

        latents, image_latents = self._prepare_latents(
            image_latents_5d, width, height, prompt_embeds.dtype,
            generator, latents,
        )
        img_shapes = self._build_img_shapes(height, width, vae_image_sizes)
        timesteps = self._prepare_timesteps(
            latents.shape[1], num_inference_steps, sigmas,
        )
        guidance = self._prepare_guidance(guidance_scale, latents.shape[0])

        with self._transformer:
            latents = self._denoise(
                latents=latents,
                image_latents=image_latents,
                timesteps=timesteps,
                prompt_embeds=prompt_embeds,
                prompt_embeds_mask=prompt_embeds_mask,
                negative_prompt_embeds=negative_prompt_embeds,
                negative_prompt_embeds_mask=negative_prompt_embeds_mask,
                img_shapes=img_shapes,
                guidance=guidance,
                attention_kwargs=attention_kwargs,
                true_cfg_scale=true_cfg_scale,
                do_true_cfg=do_true_cfg,
                on_step=on_step,
            )

        if output_type == "latent":
            return latents

        with self._vae:
            unpacked = _unpack_latents(latents, height, width)
            decoded = self._vae.pipe_decode(unpacked)
            self._device.signal(ModelInferencedEvent(self._vae.id))
        _fire(on_step)

        postprocessed = self._image_processor.postprocess(
            decoded[:, :, 0], output_type="pil",
        )
        return cast(list[Image.Image], postprocessed)

    @staticmethod
    def _validate_inputs(
        prompt: str | None,
        prompt_embeds: torch.Tensor | None,
        negative_prompt: str | None,
        negative_prompt_embeds: torch.Tensor | None,
        max_sequence_length: int,
    ) -> None:
        if max_sequence_length > 1024:
            raise ValueError(
                f"max_sequence_length cannot exceed 1024, got {max_sequence_length}",
            )
        if prompt is None and prompt_embeds is None:
            raise ValueError("Provide either `prompt` or `prompt_embeds`.")
        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                "Provide only one of `prompt` or `prompt_embeds`.")
        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                "Provide only one of `negative_prompt` or `negative_prompt_embeds`.",
            )

    def _preprocess(
        self, images: list[Image.Image],
    ) -> tuple[list[Image.Image], list[torch.Tensor], list[tuple[int, int]]]:
        condition_images: list[Image.Image] = []
        vae_images: list[torch.Tensor] = []
        vae_image_sizes: list[tuple[int, int]] = []
        for img in images:
            w, h = img.size
            ratio = w / h
            cw, ch = _calculate_dimensions(CONDITION_IMAGE_AREA, ratio)
            vw, vh = _calculate_dimensions(VAE_IMAGE_AREA, ratio)
            vae_image_sizes.append((vw, vh))
            condition_images.append(
                cast(Image.Image, self._image_processor.resize(img, ch, cw)),
            )
            vae_images.append(
                self._image_processor.preprocess(img, vh, vw).unsqueeze(2),
            )
        return condition_images, vae_images, vae_image_sizes

    def _encode_prompt(
        self, prompt: str, images: list[Image.Image],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        base_img_prompt = "".join(
            IMG_PROMPT_TEMPLATE.format(i + 1) for i in range(len(images))
        )
        txt = [PROMPT_TEMPLATE.format(base_img_prompt + prompt)]

        model_inputs = cast(Any, self._processor)(
            text=txt, images=images, padding=True, return_tensors="pt",
        ).to(DEVICE)

        hidden_states = self._text_encoder.pipe(
            input_ids=model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            pixel_values=model_inputs.pixel_values,
            image_grid_thw=model_inputs.image_grid_thw,
        )

        split_hidden = _extract_masked_hidden(
            hidden_states, model_inputs.attention_mask,
        )
        split_hidden = [e[PROMPT_TEMPLATE_START_IDX:] for e in split_hidden]
        attn_masks = [
            torch.ones(e.size(0), dtype=torch.long, device=e.device)
            for e in split_hidden
        ]
        max_seq_len = max(e.size(0) for e in split_hidden)

        prompt_embeds = torch.stack([
            torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))])
            for u in split_hidden
        ])
        prompt_embeds_mask: torch.Tensor | None = torch.stack([
            torch.cat([u, u.new_zeros(max_seq_len - u.size(0))])
            for u in attn_masks
        ])
        prompt_embeds = prompt_embeds.to(
            dtype=self._text_encoder.dtype, device=DEVICE,
        )
        if prompt_embeds_mask is not None and prompt_embeds_mask.all():
            prompt_embeds_mask = None
        return prompt_embeds, prompt_embeds_mask

    @staticmethod
    def _prepare_latents(
        image_latents_5d: list[torch.Tensor],
        width: int,
        height: int,
        dtype: torch.dtype,
        generator: torch.Generator | None,
        latents: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        latent_h = 2 * (height // PACK_FACTOR)
        latent_w = 2 * (width // PACK_FACTOR)
        shape = (1, 1, NUM_CHANNELS_LATENTS, latent_h, latent_w)

        packed = [
            _pack_latents(
                img_latents, 1, NUM_CHANNELS_LATENTS,
                img_latents.shape[3], img_latents.shape[4],
            )
            for img_latents in image_latents_5d
        ]
        image_latents = torch.cat(packed, dim=1)

        if latents is None:
            noise = randn_tensor(
                shape, generator=generator,
                device=torch.device(DEVICE), dtype=dtype,
            )
            latents = _pack_latents(
                noise, 1, NUM_CHANNELS_LATENTS, latent_h, latent_w,
            )
        else:
            latents = latents.to(device=DEVICE, dtype=dtype)

        return latents, image_latents

    @staticmethod
    def _build_img_shapes(
        height: int, width: int, vae_image_sizes: list[tuple[int, int]],
    ) -> list[list[tuple[int, int, int]]]:
        target = (1, height // PACK_FACTOR, width // PACK_FACTOR)
        per_image = [
            (1, vh // PACK_FACTOR, vw // PACK_FACTOR)
            for vw, vh in vae_image_sizes
        ]
        return [[target, *per_image]]

    def _prepare_timesteps(
        self,
        image_seq_len: int,
        num_inference_steps: int,
        sigmas: list[float] | None,
    ) -> torch.Tensor:
        if sigmas is None:
            sigmas = np.linspace(
                1.0, 1 / num_inference_steps, num_inference_steps,
            ).tolist()
        cfg = self._scheduler.config
        mu = _calculate_shift(
            image_seq_len,
            cfg.get("base_image_seq_len", 256),
            cfg.get("max_image_seq_len", 4096),
            cfg.get("base_shift", 0.5),
            cfg.get("max_shift", 1.15),
        )
        self._scheduler.set_timesteps(
            num_inference_steps, device=DEVICE, sigmas=sigmas, mu=mu,
        )
        return cast(torch.Tensor, self._scheduler.timesteps)

    @staticmethod
    def _prepare_guidance(
        guidance_scale: float | None, batch_size: int,
    ) -> torch.Tensor | None:
        if guidance_scale is None:
            return None
        g = torch.full([1], guidance_scale, device=DEVICE, dtype=torch.float32)
        return g.expand(batch_size)

    def _denoise(
        self,
        *,
        latents: torch.Tensor,
        image_latents: torch.Tensor,
        timesteps: torch.Tensor,
        prompt_embeds: torch.Tensor,
        prompt_embeds_mask: torch.Tensor | None,
        negative_prompt_embeds: torch.Tensor | None,
        negative_prompt_embeds_mask: torch.Tensor | None,
        img_shapes: list[list[tuple[int, int, int]]],
        guidance: torch.Tensor | None,
        attention_kwargs: dict[str, Any] | None,
        true_cfg_scale: float,
        do_true_cfg: bool,
        on_step: OnStep | None,
    ) -> torch.Tensor:
        self._scheduler.set_begin_index(0)
        for t in timesteps:
            latent_model_input = torch.cat([latents, image_latents], dim=1)
            timestep = t.expand(latents.shape[0]).to(latents.dtype)

            with self._transformer.cache_context("cond"):
                noise_pred = self._transformer.pipe(
                    hidden_states=latent_model_input,
                    timestep=timestep / 1000,
                    guidance=guidance,
                    encoder_hidden_states=prompt_embeds,
                    encoder_hidden_states_mask=prompt_embeds_mask,
                    img_shapes=img_shapes,
                    attention_kwargs=attention_kwargs,
                )
            self._device.signal(ModelInferencedEvent(self._transformer.id))
            noise_pred = noise_pred[:, : latents.size(1)]

            if do_true_cfg:
                assert negative_prompt_embeds is not None
                with self._transformer.cache_context("uncond"):
                    neg_noise_pred = self._transformer.pipe(
                        hidden_states=latent_model_input,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        encoder_hidden_states=negative_prompt_embeds,
                        encoder_hidden_states_mask=negative_prompt_embeds_mask,
                        img_shapes=img_shapes,
                        attention_kwargs=attention_kwargs,
                    )
                self._device.signal(ModelInferencedEvent(self._transformer.id))
                neg_noise_pred = neg_noise_pred[:, : latents.size(1)]
                comb_pred = neg_noise_pred + true_cfg_scale * (
                    noise_pred - neg_noise_pred
                )
                cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
                noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
                noise_pred = comb_pred * (cond_norm / noise_norm)

            latents_dtype = latents.dtype
            stepped = cast(Any, self._scheduler).step(
                noise_pred, t, latents, return_dict=False,
            )
            latents = cast(torch.Tensor, stepped[0])
            if latents.dtype != latents_dtype:
                latents = latents.to(latents_dtype)

            _fire(on_step)

        return latents
