from __future__ import annotations

import asyncio
import signal
import threading
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import structlog
from botocore.config import Config as BotocoreConfig
from structlog.stdlib import BoundLogger

from src import metrics
from src.app.add import AddService
from src.app.animate import AnimateService
from src.app.cancellation import CancellationService
from src.app.delete import DeleteService
from src.app.norm_map import NormMapService
from src.app.rmbg import RmbgService
from src.app.stand import StandService
from src.app.ticket import TicketService
from src.config import ServiceConfigV1
from src.config.service_v1 import (
    AwsStorageProvider,
    KafkaCancellationProvider,
    KafkaTaskProvider,
    LokiLogProvider,
    PrometheusMetricsProvider,
    RestCancellationProvider as RestCancellationProviderConfig,
)
from src.controller.cancellation.kafka.config import KafkaCancellationConfig
from src.controller.cancellation.kafka.KafkaCancellationController import (
    KafkaCancellationController,
)
from src.controller.cancellation.rest import (
    RestCancellationConfig,
    RestCancellationProvider,
)
from src.controller.cancellation.rest.config import (
    RestCancellationAuthorization,
    RestCancellationCertificate,
)
from src.controller.task.kafka.config import KafkaControllerConfig
from src.controller.task.kafka.KafkaTaskController import KafkaTaskController
from src.domain.pipeline import (
    InspyreNetRmbgPipeline,
    QwenImageEditAddPipeline,
    QwenImageEditDeletePipeline,
    QwenImageEditStandPipeline,
    StableNormal01NormalMapPipeline,
    Wan22AnimatePipeline,
)
from src.domain.pipeline.model.InSPyReNetBackboneModel import InSPyReNetBackboneModel
from src.domain.pipeline.model.InSPyReNetDecoderModel import InSPyReNetDecoderModel
from src.domain.pipeline.model.StableNormalModel import StableNormalModel
from src.domain.worker.device.Device import Device
from src.domain.worker.device.eviction.LruEvictionPolicy import LruEvictionPolicy
from src.domain.worker.scheduler.Scheduler import Scheduler
from src.domain.worker.Worker import Worker
from src.infra.loki import LokiSink, LokiSinkConfig
from src.infra.loki.LokiSink import open_sinks
from src.infra.prometheus import PrometheusServer
from src.infra.s3 import AwsStorage, AwsStorageConfig
from src.IO import IO
from src.repository.s3 import S3StorageRepository

log: BoundLogger = structlog.get_logger(__name__)


def _build_botocore_config(provider: AwsStorageProvider) -> BotocoreConfig | None:
    cfg = provider.config
    if cfg is None:
        return None
    kwargs: dict[str, Any] = {}
    for key in (
        "connect_timeout", "read_timeout", "max_pool_connections",
        "signature_version", "region_name", "user_agent_extra",
    ):
        value = getattr(cfg, key)
        if value is not None:
            kwargs[key] = value
    if cfg.retries is not None:
        retries: dict[str, Any] = {}
        if cfg.retries.max_attempts is not None:
            retries["max_attempts"] = cfg.retries.max_attempts
        if cfg.retries.mode is not None:
            retries["mode"] = cfg.retries.mode
        if retries:
            kwargs["retries"] = retries
    s3_kwargs: dict[str, Any] = {}
    if cfg.addressing_style is not None:
        s3_kwargs["addressing_style"] = cfg.addressing_style
    if cfg.use_accelerate_endpoint is not None:
        s3_kwargs["use_accelerate_endpoint"] = cfg.use_accelerate_endpoint
    if cfg.use_dualstack_endpoint is not None:
        s3_kwargs["use_dualstack_endpoint"] = cfg.use_dualstack_endpoint
    if cfg.payload_signing_enabled is not None:
        s3_kwargs["payload_signing_enabled"] = cfg.payload_signing_enabled
    if s3_kwargs:
        kwargs["s3"] = s3_kwargs
    if not kwargs:
        return None
    return BotocoreConfig(**kwargs)


def _build_storage_config(provider: AwsStorageProvider) -> AwsStorageConfig:
    creds = provider.credentials
    config: AwsStorageConfig = {
        "service_name": provider.service_name,
        "bucket": provider.bucket,
        "region_name": provider.region_name,
        "endpoint_url": provider.endpoint_url,
        "api_version": provider.api_version,
        "use_ssl": provider.use_ssl,
        "verify": provider.verify,
        "profile_name": creds.profile_name if creds else None,
        "aws_access_key_id": creds.aws_access_key_id if creds else None,
        "aws_secret_access_key": creds.aws_secret_access_key if creds else None,
        "aws_session_token": creds.aws_session_token if creds else None,
        "botocore_config": _build_botocore_config(provider),
    }
    return config


def _kafka_bootstrap_servers(value: str | list[str]) -> str:
    if isinstance(value, list):
        return ",".join(value)
    return value


@asynccontextmanager
async def _open_rest_cancellation(
    providers: list[RestCancellationProvider],
) -> AsyncIterator[list[RestCancellationProvider]]:
    entered: list[RestCancellationProvider] = []
    try:
        for provider in providers:
            await provider.__aenter__()
            entered.append(provider)
        yield entered
    finally:
        for provider in reversed(entered):
            await provider.__aexit__(None, None, None)


def _build_rest_cancellation_config(
    provider: RestCancellationProviderConfig,
) -> RestCancellationConfig:
    auth: RestCancellationAuthorization | None = None
    if provider.authorization is not None:
        cert: RestCancellationCertificate | None = None
        if provider.authorization.certificate is not None:
            cert = RestCancellationCertificate(
                cert=provider.authorization.certificate.cert,
                key=provider.authorization.certificate.key,
            )
        auth = RestCancellationAuthorization(
            header=provider.authorization.header,
            certificate=cert,
        )
    return RestCancellationConfig(
        path=provider.path,
        method=provider.method,
        authorization=auth,
        timeout_seconds=provider.timeout_seconds,
        verify_ssl=provider.verify_ssl,
    )


def _build_loki_config(provider: LokiLogProvider) -> LokiSinkConfig:
    return LokiSinkConfig(
        url=provider.url,
        endpoint=provider.endpoint,
        tenant_id=provider.tenant_id,
        auth=(provider.auth.username, provider.auth.password) if provider.auth else None,
        labels=dict(provider.labels),
        timeout_seconds=provider.timeout_seconds if provider.timeout_seconds is not None else 5.0,
        batch_size=provider.batch_size if provider.batch_size is not None else 100,
        batch_interval_seconds=provider.batch_interval_seconds if provider.batch_interval_seconds is not None else 1.0,
        retries=provider.retries if provider.retries is not None else 3,
        retry_backoff_seconds=provider.retry_backoff_seconds if provider.retry_backoff_seconds is not None else 0.5,
        verify_ssl=provider.verify_ssl if provider.verify_ssl is not None else True,
    )


class Service:
    def __init__(self, config: ServiceConfigV1):
        self._version = config.version

        self._loki_sinks: list[LokiSink] = [
            LokiSink(_build_loki_config(provider))
            for provider in config.logs
            if isinstance(provider, LokiLogProvider)
        ]
        self._prometheus_servers: list[PrometheusServer] = []
        prometheus_namespace = ""
        prometheus_labels: dict[str, str] = {}
        for provider in config.metrics:
            if isinstance(provider, PrometheusMetricsProvider):
                self._prometheus_servers.append(PrometheusServer(
                    host=provider.host or "0.0.0.0",
                    port=provider.port or 9100,
                ))
                if provider.namespace:
                    prometheus_namespace = provider.namespace
                prometheus_labels.update(provider.labels)
        metrics.configure(namespace=prometheus_namespace, labels=prometheus_labels)

        scheduler = Scheduler()
        device = Device(
            eviction_policy=LruEvictionPolicy(),
            gpu_mem=int(config.device.gpu_mem),
        )

        self._storage = S3StorageRepository(
            prefix=config.storage.prefix,
            config=_build_storage_config(config.storage),
        )

        ticket = TicketService(get_ticket=scheduler.queue)

        inspyre_backbone = InSPyReNetBackboneModel(
            device=device, model_dir=config.models.dir)
        inspyre_decoder = InSPyReNetDecoderModel(
            device=device, model_dir=config.models.dir)
        stable_normal = StableNormalModel(
            device=device, model_dir=config.models.dir)

        self._qwen_image_edit_add = QwenImageEditAddPipeline(
            device=device,
            model_dir=config.models.dir,
            hf_path=config.capabilities.add.hf_path,
        )
        self._qwen_image_edit_delete = QwenImageEditDeletePipeline(
            device=device,
            model_dir=config.models.dir,
            hf_path=config.capabilities.delete.hf_path,
        )
        self._qwen_image_edit_stand = QwenImageEditStandPipeline(
            device=device,
            model_dir=config.models.dir,
            hf_path=config.capabilities.stand.hf_path,
        )
        self._wan22_animate = Wan22AnimatePipeline(
            device=device,
            model_dir=config.models.dir,
            hf_path=config.capabilities.animate.hf_path,
        )
        self._inspyre_net_rmbg = InspyreNetRmbgPipeline(
            device=device,
            backbone=inspyre_backbone,
            decoder=inspyre_decoder,
            batch_size=4,
            hf_path=config.capabilities.rmbg.hf_path,
        )
        self._stable_normal_normal_map = StableNormal01NormalMapPipeline(
            device=device,
            model=stable_normal,
            batch_size=4,
            hf_path=config.capabilities.norm_map.hf_path,
        )

        self._add = AddService(
            get_character=self._storage.get_character,
            save_character=self._storage.save_character,
            pipeline=self._qwen_image_edit_add,
        )
        self._delete = DeleteService(
            get_character=self._storage.get_character,
            save_character=self._storage.save_character,
            pipeline=self._qwen_image_edit_delete,
        )
        self._stand = StandService(
            get_character=self._storage.get_character,
            save_character=self._storage.save_character,
            pipeline=self._qwen_image_edit_stand,
        )
        self._animate = AnimateService(
            get_character=self._storage.get_character,
            save_frames=self._storage.save_frames,
            pipeline=self._wan22_animate,
        )
        self._rmbg = RmbgService(
            get_frames=self._storage.get_frames,
            save_frames=self._storage.save_frames,
            pipeline=self._inspyre_net_rmbg,
        )
        self._norm_map = NormMapService(
            get_frames=self._storage.get_frames,
            save_frames=self._storage.save_frames,
            pipeline=self._stable_normal_normal_map,
        )

        self._rest_cancellation_providers: list[RestCancellationProvider] = []
        kafka_cancellation_providers: list[KafkaCancellationProvider] = []
        for provider in config.cancellation:
            if isinstance(provider, KafkaCancellationProvider):
                kafka_cancellation_providers.append(provider)
            elif isinstance(provider, RestCancellationProviderConfig):
                self._rest_cancellation_providers.append(
                    RestCancellationProvider(
                        _build_rest_cancellation_config(provider)))

        self._cancellation = CancellationService(
            get_task_cancelled=[
                p.get_task_cancelled for p in self._rest_cancellation_providers
            ],
        )

        self._kafka_cancellation_controllers: list[KafkaCancellationController] = [
            self._build_kafka_cancellation_controller(provider)
            for provider in kafka_cancellation_providers
        ]

        self._task_controllers: list[KafkaTaskController] = []
        for task_provider in config.tasks:
            if isinstance(task_provider, KafkaTaskProvider):
                self._task_controllers.append(self._build_kafka_controller(
                    task_provider, ticket,
                ))
            else:
                log.warning(
                    "Task provider type not supported, ignoring",
                    type=getattr(task_provider, "type", "<unknown>"),
                )

        self._io = IO()
        self._worker = Worker(scheduler=scheduler, device=device)

    def _build_kafka_cancellation_controller(
        self,
        provider: KafkaCancellationProvider,
    ) -> KafkaCancellationController:
        return KafkaCancellationController(
            config=KafkaCancellationConfig(
                bootstrap_servers=_kafka_bootstrap_servers(
                    provider.connection.bootstrap_servers),
                client_id=provider.connection.client_id or "worker-service",
                group_id=provider.consumer.group_id,
                topics=provider.consumer.topics,
            ),
            cancel=self._cancellation,
        )

    def _build_kafka_controller(
        self,
        provider: KafkaTaskProvider,
        ticket: TicketService,
    ) -> KafkaTaskController:
        return KafkaTaskController(
            config=KafkaControllerConfig(
                bootstrap_servers=_kafka_bootstrap_servers(
                    provider.connection.bootstrap_servers),
                client_id=provider.connection.client_id or "worker-service",
                group_id=provider.consumer.group_id,
                topics=provider.consumer.topics,
                produce_topic=provider.publisher.topic,
            ),
            cancel=self._cancellation,
            ticket=ticket,
            add=self._add,
            delete=self._delete,
            stand=self._stand,
            animate=self._animate,
            rmbg=self._rmbg,
            norm_map=self._norm_map,
        )

    @property
    def loki_sinks(self) -> list[LokiSink]:
        return list(self._loki_sinks)

    async def _io_main(self) -> None:
        async with (
            self._storage,
            open_sinks(self._loki_sinks),
            _open_rest_cancellation(self._rest_cancellation_providers),
        ):
            controllers = [*self._task_controllers,
                           *self._kafka_cancellation_controllers]
            tasks = [asyncio.create_task(c.run()) for c in controllers]
            try:
                await asyncio.gather(*tasks)
            except asyncio.CancelledError:
                for t in tasks:
                    t.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise

    def do(self) -> None:
        log.info("Service starting...",
                 task_name="Startup Service", version=self._version)

        for server in self._prometheus_servers:
            server.start()

        done = threading.Event()

        signal.signal(signal.SIGTERM, lambda *_: done.set())
        signal.signal(signal.SIGINT, lambda *_: done.set())

        def run_worker():
            try:
                self._worker.start()
            except Exception:
                log.exception("Worker failed")
            finally:
                done.set()

        worker_thread = threading.Thread(target=run_worker)
        worker_thread.start()

        def run_io():
            try:
                self._io.start([self._io_main()])
            except Exception:
                log.exception("IO failed")
            finally:
                done.set()

        io_thread = threading.Thread(target=run_io)
        io_thread.start()

        log.info("Service started.")

        done.wait()

        log.info("Shutting down...", task_name="Shutdown Service")

        self._io.stop()
        io_thread.join()

        self._worker.stop()
        worker_thread.join()

        for server in self._prometheus_servers:
            server.stop()

        log.info("Service stopped.")
