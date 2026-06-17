import abc
from dataclasses import dataclass
from typing import Protocol

import structlog
from structlog.stdlib import BoundLogger

from aiokafka import AIOKafkaConsumer, TopicPartition


log: BoundLogger = structlog.get_logger(__name__)


class Commitable(Protocol):
    topic: str
    partition: int
    offset: int


@dataclass
class KafkaMessage:
    topic: str
    partition: int
    offset: int
    key: bytes | None
    value: bytes


@dataclass
class KafkaConsumer(abc.ABC):
    def __init__(
        self,
        bootstrap_servers: str,
        client_id: str,
        group_id: str,
        topics: list[str],
    ):
        super().__init__()

        self._consumer = AIOKafkaConsumer(
            *topics,
            bootstrap_servers=bootstrap_servers,
            group_id=group_id,
            client_id=client_id,
            enable_auto_commit=False,
            auto_offset_reset="earliest",
        )

    async def run(self):
        log.info("Consumer loop started")

        await self._consumer.start()
        try:
            async for record in self._consumer:
                log.info(
                    "Message received",
                    topic=record.topic,
                    partition=record.partition,
                    offset=record.offset,
                )

                if record.value is not None:
                    await self.handle(KafkaMessage(
                        topic=record.topic,
                        partition=record.partition,
                        offset=record.offset,
                        key=record.key,
                        value=record.value,
                    ))
                else:
                    await self.commit(record)
        finally:
            await self._consumer.stop()

        log.info("Consumer loop finished")

    async def commit(self, message: Commitable):
        await self._consumer.commit({TopicPartition(message.topic, message.partition): message.offset + 1})

    async def handle(self, message: KafkaMessage):
        ...
