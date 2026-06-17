from dataclasses import dataclass


@dataclass
class KafkaCancellationConfig:
    bootstrap_servers: str
    client_id: str
    group_id: str
    topics: list[str]
