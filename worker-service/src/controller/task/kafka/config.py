from dataclasses import dataclass


@dataclass
class KafkaControllerConfig:
    bootstrap_servers: str
    client_id: str
    group_id: str
    topics: list[str]
    produce_topic: str
