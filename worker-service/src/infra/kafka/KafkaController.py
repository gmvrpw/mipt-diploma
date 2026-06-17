from dataclasses import dataclass

import structlog
from structlog.stdlib import BoundLogger

from aiokafka import AIOKafkaProducer, TopicPartition

from .KafkaConsumer import KafkaConsumer, KafkaMessage


log: BoundLogger = structlog.get_logger(__name__)


@dataclass
class KafkaController(KafkaConsumer):
    def __init__(
        self,
        bootstrap_servers: str,
        client_id: str,
        group_id: str,
        consume_topics: list[str],
        produce_topic: str,
    ):
        super().__init__(
            bootstrap_servers=bootstrap_servers,
            client_id=client_id,
            group_id=group_id,
            topics=consume_topics,
        )

        self._producer = AIOKafkaProducer(
            bootstrap_servers=bootstrap_servers,
            acks=0,
        )

        self._consumer_group_id = group_id
        self._produce_topic = produce_topic

    async def run(self):
        log.info("Controller started")

        await self._producer.start()
        try:
            await super().run()
        finally:
            await self._producer.stop()

        log.info("Controller finished")

    async def response(self, message: KafkaMessage, value: bytes):
        async with self._producer.transaction():
            await self._producer.send(topic=self._produce_topic, value=value)
            await self._producer.send_offsets_to_transaction(
                {TopicPartition(message.topic, message.partition): message.offset + 1},
                self._consumer_group_id,
            )
