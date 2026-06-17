from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BeforeValidator, ByteSize, Field, ValidationError

from ._common import Duration, _Strict, format_errors


_DEFAULT_GPU_MEM = 80 * 1024 ** 3


def _expand_retries(value: Any) -> Any:
    if isinstance(value, int) and not isinstance(value, bool):
        return {"read": value, "write": value}
    return value


def _expand_timeout(value: Any) -> Any:
    if isinstance(value, str):
        return {"read": value, "write": value}
    return value


class ReadWriteRetries(_Strict):
    read: int = Field(ge=0)
    write: int = Field(ge=0)


class ReadWriteTimeouts(_Strict):
    read: Duration
    write: Duration


class _QwenImageEditBase(_Strict):
    hf_path: str = "Qwen/Qwen-Image-Edit-2509"


class QwenImageEditAddPipeline(_QwenImageEditBase):
    type: Literal["QwenImageEditAddPipeline"] = "QwenImageEditAddPipeline"


class QwenImageEditDeletePipeline(_QwenImageEditBase):
    type: Literal["QwenImageEditDeletePipeline"] = "QwenImageEditDeletePipeline"


class QwenImageEditStandPipeline(_QwenImageEditBase):
    type: Literal["QwenImageEditStandPipeline"] = "QwenImageEditStandPipeline"


class Wan22AnimatePipeline(_Strict):
    type: Literal["Wan22AnimatePipeline"] = "Wan22AnimatePipeline"
    hf_path: str = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"


class InspyreNetRmbgPipeline(_Strict):
    type: Literal["InspyreNetRmbgPipeline"] = "InspyreNetRmbgPipeline"
    hf_path: str = "plemeri/InSPyReNet"


class StableNormal01NormalMapPipeline(_Strict):
    type: Literal["StableNormal01NormalMapPipeline"] = "StableNormal01NormalMapPipeline"
    hf_path: str = "Stable-X/stable-normal-v0-1"


class ModelsConfig(_Strict):
    dir: str = "models"


class DeviceConfig(_Strict):
    gpu_mem: ByteSize = Field(default=_DEFAULT_GPU_MEM)


class CapabilitiesConfig(_Strict):
    add: QwenImageEditAddPipeline = Field(default_factory=QwenImageEditAddPipeline)
    delete: QwenImageEditDeletePipeline = Field(default_factory=QwenImageEditDeletePipeline)
    stand: QwenImageEditStandPipeline = Field(default_factory=QwenImageEditStandPipeline)
    animate: Wan22AnimatePipeline = Field(default_factory=Wan22AnimatePipeline)
    rmbg: InspyreNetRmbgPipeline = Field(default_factory=InspyreNetRmbgPipeline)
    norm_map: StableNormal01NormalMapPipeline = Field(default_factory=StableNormal01NormalMapPipeline)


class OutputsConfig(_Strict):
    dir: str = "outputs"
    retries: Annotated[ReadWriteRetries, BeforeValidator(_expand_retries)] = Field(
        default_factory=lambda: ReadWriteRetries(read=3, write=3)
    )
    timeout: Annotated[ReadWriteTimeouts, BeforeValidator(_expand_timeout)] = Field(
        default_factory=lambda: ReadWriteTimeouts(read=1.0, write=1.0)
    )


class StandaloneConfigV1(_Strict):
    version: Literal["v1"] = "v1"
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    device: DeviceConfig = Field(default_factory=DeviceConfig)
    capabilities: CapabilitiesConfig = Field(default_factory=CapabilitiesConfig)
    outputs: OutputsConfig = Field(default_factory=OutputsConfig)


def _resolve(value: str, base: Path) -> str:
    if os.path.isabs(value):
        return value
    return os.path.normpath(os.path.join(base, value))


def _resolve_paths(config: StandaloneConfigV1, base: Path) -> StandaloneConfigV1:
    return config.model_copy(
        update={
            "models": config.models.model_copy(
                update={"dir": _resolve(config.models.dir, base)}
            ),
            "outputs": config.outputs.model_copy(
                update={"dir": _resolve(config.outputs.dir, base)}
            ),
        }
    )


def parse(data: Any, *, base_path: Path) -> StandaloneConfigV1:
    from . import ConfigError

    try:
        config = StandaloneConfigV1.model_validate(data)
    except ValidationError as e:
        raise ConfigError(
            f"invalid standalone-v1 config:\n{format_errors(e)}"
        ) from e

    return _resolve_paths(config, base_path)
