from pathlib import Path

import pytest
import yaml

from src.config import ConfigError, parse_service_config
from src.config.service_v1 import (
    AwsStorageProvider,
    KafkaCancellationProvider,
    KafkaTaskProvider,
    LokiLogProvider,
    PrometheusMetricsProvider,
    RestCancellationProvider,
)


def _write(tmp_path: Path, data: dict) -> str:
    p = tmp_path / "service.yml"
    p.write_text(yaml.safe_dump(data))
    return str(p)


_MIN_STORAGE = {"type": "aws", "bucket": "bk"}


def test_minimal_config(tmp_path: Path) -> None:
    path = _write(tmp_path, {
        "version": "v1",
        "storage": _MIN_STORAGE,
    })
    config = parse_service_config(path, base_path=tmp_path)
    assert isinstance(config.storage, AwsStorageProvider)
    assert config.storage.bucket == "bk"
    assert config.tasks == []
    assert config.cancellation == []


def test_full_config_parses_all_providers(tmp_path: Path) -> None:
    path = _write(tmp_path, {
        "version": "v1",
        "storage": {**_MIN_STORAGE, "prefix": "dev"},
        "tasks": [{
            "type": "kafka",
            "connection": {"bootstrap_servers": "kafka:9092"},
            "consumer": {"group_id": "g", "topics": ["t"]},
            "publisher": {"topic": "out"},
        }],
        "cancellation": [
            {
                "type": "kafka",
                "connection": {"bootstrap_servers": "kafka:9092"},
                "consumer": {"group_id": "gc", "topics": ["cancel"]},
            },
            {
                "type": "rest",
                "path": "https://api/tasks/{task_id}/cancelled",
                "method": "POST",
                "authorization": {"header": "X-Token: abc"},
            },
        ],
        "logs": [{"type": "loki", "url": "http://loki"}],
        "metrics": [{"type": "prometheus", "port": 9100, "namespace": "ws"}],
    })
    config = parse_service_config(path, base_path=tmp_path)
    assert len(config.tasks) == 1 and isinstance(config.tasks[0], KafkaTaskProvider)
    assert len(config.cancellation) == 2
    assert isinstance(config.cancellation[0], KafkaCancellationProvider)
    assert isinstance(config.cancellation[1], RestCancellationProvider)
    assert config.cancellation[1].method == "POST"
    assert isinstance(config.logs[0], LokiLogProvider)
    assert isinstance(config.metrics[0], PrometheusMetricsProvider)
    assert config.metrics[0].namespace == "ws"


def test_storage_required(tmp_path: Path) -> None:
    path = _write(tmp_path, {"version": "v1"})
    with pytest.raises(ConfigError) as exc:
        parse_service_config(path, base_path=tmp_path)
    assert "storage" in str(exc.value)


def test_discriminator_unknown_type(tmp_path: Path) -> None:
    path = _write(tmp_path, {
        "version": "v1",
        "storage": _MIN_STORAGE,
        "tasks": [{"type": "rabbit"}],
    })
    with pytest.raises(ConfigError) as exc:
        parse_service_config(path, base_path=tmp_path)
    assert "tasks.0" in str(exc.value)


def test_rest_authorization_certificate(tmp_path: Path) -> None:
    path = _write(tmp_path, {
        "version": "v1",
        "storage": _MIN_STORAGE,
        "cancellation": [{
            "type": "rest",
            "path": "https://x/{task_id}",
            "authorization": {"certificate": {"cert": "/c", "key": "/k"}},
        }],
    })
    config = parse_service_config(path, base_path=tmp_path)
    rest = config.cancellation[0]
    assert isinstance(rest, RestCancellationProvider)
    assert rest.authorization is not None
    assert rest.authorization.certificate is not None
    assert rest.authorization.certificate.cert == "/c"
    assert rest.authorization.certificate.key == "/k"
