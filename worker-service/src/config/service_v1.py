from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import ByteSize, Field, ValidationError

from ._common import Duration, _Strict, format_errors


_DEFAULT_GPU_MEM = 80 * 1024 ** 3


class ModelsConfig(_Strict):
    dir: str = "models"


class DeviceConfig(_Strict):
    gpu_mem: ByteSize = Field(default=_DEFAULT_GPU_MEM)


# ---------- Pipelines (same as standalone) ----------


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


class CapabilitiesConfig(_Strict):
    add: QwenImageEditAddPipeline = Field(default_factory=QwenImageEditAddPipeline)
    delete: QwenImageEditDeletePipeline = Field(default_factory=QwenImageEditDeletePipeline)
    stand: QwenImageEditStandPipeline = Field(default_factory=QwenImageEditStandPipeline)
    animate: Wan22AnimatePipeline = Field(default_factory=Wan22AnimatePipeline)
    rmbg: InspyreNetRmbgPipeline = Field(default_factory=InspyreNetRmbgPipeline)
    norm_map: StableNormal01NormalMapPipeline = Field(default_factory=StableNormal01NormalMapPipeline)


# ---------- Task providers ----------


class KafkaConnectionConfig(_Strict):
    bootstrap_servers: str | list[str]
    client_id: str | None = None
    security_protocol: str | None = None
    sasl_mechanism: str | None = None
    sasl_username: str | None = None
    sasl_password: str | None = None
    ssl_cafile: str | None = None
    ssl_certfile: str | None = None
    ssl_keyfile: str | None = None
    request_timeout_ms: int | None = None
    connections_max_idle_ms: int | None = None
    retries: int | None = None
    retry_backoff_ms: int | None = None


class KafkaConsumerConfig(_Strict):
    group_id: str
    topics: list[str]
    session_timeout_ms: int | None = None
    heartbeat_interval_ms: int | None = None
    fetch_min_bytes: int | None = None
    fetch_max_wait_ms: int | None = None


class KafkaPublisherConfig(_Strict):
    topic: str
    compression_type: str | None = None
    linger_ms: int | None = None
    batch_size: int | None = None
    max_request_size: int | None = None
    delivery_timeout_ms: int | None = None


class KafkaTaskProvider(_Strict):
    type: Literal["kafka"]
    connection: KafkaConnectionConfig
    consumer: KafkaConsumerConfig
    publisher: KafkaPublisherConfig


TaskProvider = Annotated[
    KafkaTaskProvider,
    Field(discriminator="type"),
]


# ---------- Log providers ----------


class LokiAuth(_Strict):
    username: str
    password: str


class LokiLogProvider(_Strict):
    type: Literal["loki"]
    url: str
    endpoint: str = "/loki/api/v1/push"
    tenant_id: str | None = None
    auth: LokiAuth | None = None
    labels: dict[str, str] = Field(default_factory=dict)
    timeout_seconds: float | None = None
    batch_size: int | None = None
    batch_interval_seconds: float | None = None
    retries: int | None = None
    retry_backoff_seconds: float | None = None
    verify_ssl: bool | None = None


LogProvider = Annotated[
    LokiLogProvider,
    Field(discriminator="type"),
]


# ---------- Metrics providers ----------


class PrometheusMetricsProvider(_Strict):
    type: Literal["prometheus"]
    host: str | None = None
    port: int | None = None
    path: str | None = None
    namespace: str | None = None
    labels: dict[str, str] = Field(default_factory=dict)


MetricsProvider = Annotated[
    PrometheusMetricsProvider,
    Field(discriminator="type"),
]


# ---------- Storage providers ----------


class AwsCredentials(_Strict):
    profile_name: str | None = None
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None
    aws_session_token: str | None = None


class AwsBotocoreRetries(_Strict):
    max_attempts: int | None = None
    mode: str | None = None


class AwsBotocoreConfig(_Strict):
    connect_timeout: float | None = None
    read_timeout: float | None = None
    max_pool_connections: int | None = None
    signature_version: str | None = None
    region_name: str | None = None
    user_agent_extra: str | None = None
    retries: AwsBotocoreRetries | None = None
    addressing_style: str | None = None
    use_accelerate_endpoint: bool | None = None
    use_dualstack_endpoint: bool | None = None
    payload_signing_enabled: bool | None = None


class AwsTransferConfig(_Strict):
    multipart_threshold: int | None = None
    multipart_chunksize: int | None = None
    max_concurrency: int | None = None
    max_io_queue: int | None = None
    io_chunksize: int | None = None
    use_threads: bool | None = None


class AwsStorageProvider(_Strict):
    type: Literal["aws"]
    name: str | None = None
    service_name: str = "s3"
    bucket: str
    prefix: str = ""
    region_name: str | None = None
    endpoint_url: str | None = None
    api_version: str | None = None
    use_ssl: bool | None = None
    verify: bool | None = None
    credentials: AwsCredentials | None = None
    config: AwsBotocoreConfig | None = None
    transfer: AwsTransferConfig | None = None


StorageProvider = Annotated[
    AwsStorageProvider,
    Field(discriminator="type"),
]


# ---------- Cancellation providers ----------


class KafkaCancellationProvider(_Strict):
    type: Literal["kafka"]
    connection: KafkaConnectionConfig
    consumer: KafkaConsumerConfig


class RestCertificate(_Strict):
    cert: str
    key: str


class RestAuthorization(_Strict):
    header: str | None = None
    certificate: RestCertificate | None = None


class RestCancellationProvider(_Strict):
    type: Literal["rest"]
    path: str
    method: Literal["GET", "POST"] = "GET"
    authorization: RestAuthorization | None = None
    timeout_seconds: float = 5.0
    verify_ssl: bool = True


CancellationProvider = Annotated[
    KafkaCancellationProvider | RestCancellationProvider,
    Field(discriminator="type"),
]


# ---------- Root ----------


class ServiceConfigV1(_Strict):
    version: Literal["v1"] = "v1"
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    device: DeviceConfig = Field(default_factory=DeviceConfig)
    capabilities: CapabilitiesConfig = Field(default_factory=CapabilitiesConfig)
    storage: StorageProvider
    tasks: list[TaskProvider] = Field(default_factory=list)
    cancellation: list[CancellationProvider] = Field(default_factory=list)
    logs: list[LogProvider] = Field(default_factory=list)
    metrics: list[MetricsProvider] = Field(default_factory=list)


def _resolve(value: str, base: Path) -> str:
    if os.path.isabs(value):
        return value
    return os.path.normpath(os.path.join(base, value))


def _resolve_paths(config: ServiceConfigV1, base: Path) -> ServiceConfigV1:
    return config.model_copy(
        update={
            "models": config.models.model_copy(
                update={"dir": _resolve(config.models.dir, base)}
            ),
        }
    )


def parse(data: Any, *, base_path: Path) -> ServiceConfigV1:
    from . import ConfigError

    try:
        config = ServiceConfigV1.model_validate(data)
    except ValidationError as e:
        raise ConfigError(
            f"invalid service-v1 config:\n{format_errors(e)}"
        ) from e

    return _resolve_paths(config, base_path)
