import worker_service_proto.task as proto

import structlog
from structlog.contextvars import bound_contextvars
from structlog.stdlib import BoundLogger

from src.app import CancellationService

from src.infra.kafka import KafkaConsumer, KafkaMessage

from .config import KafkaCancellationConfig
from .mapping import to_cancel_task_request

log: BoundLogger = structlog.get_logger(__name__)


class KafkaCancellationController(KafkaConsumer):
    def __init__(self, config: KafkaCancellationConfig, cancel: CancellationService):
        super().__init__(
            bootstrap_servers=config.bootstrap_servers,
            client_id=config.client_id,
            group_id=config.group_id,
            topics=config.topics,
        )

        self._cancel = cancel

    async def handle(self, message: KafkaMessage):
        try:
            request = proto.Cancelled.parse(message.value)

            with bound_contextvars(task_id=request.task_id):
                self._cancel(to_cancel_task_request(request))
        finally:
            await self.commit(message)
