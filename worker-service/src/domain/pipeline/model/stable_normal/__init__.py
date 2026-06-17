import sys

# Stable-Normal HF repo ships remote code that imports the old path
# `diffusers.models.controlnet`; current diffusers moved it to
# `diffusers.models.controlnets.controlnet`. Alias before `trust_remote_code`
# pulls the remote module.
import diffusers.models.controlnets.controlnet as _controlnet_mod
sys.modules.setdefault("diffusers.models.controlnet", _controlnet_mod)

from .pipeline_stablenormal import StableNormalPipeline, StableNormalOutput
from .pipeline_yoso_normal import YOSONormalsPipeline, YosoNormalsOutput
from .scheduler.heuristics_ddimsampler import HEURI_DDIMScheduler

__all__ = [
    "HEURI_DDIMScheduler",
    "StableNormalOutput",
    "StableNormalPipeline",
    "YOSONormalsPipeline",
    "YosoNormalsOutput",
]