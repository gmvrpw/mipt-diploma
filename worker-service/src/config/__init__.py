from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable

import structlog
import yaml
from structlog.contextvars import bound_contextvars
from structlog.stdlib import BoundLogger

from . import service_v1, standalone_v1
from .service_v1 import ServiceConfigV1
from .standalone_v1 import StandaloneConfigV1

log: BoundLogger = structlog.get_logger(__name__)

DEFAULT_CONFIG_VERSION = "v1"


class ConfigError(Exception):
    """Raised when a config file cannot be parsed or fails validation."""


_STANDALONE_PARSERS: dict[str, Callable[..., Any]] = {
    "v1": standalone_v1.parse,
}

_SERVICE_PARSERS: dict[str, Callable[..., Any]] = {
    "v1": service_v1.parse,
}


def _entrypoint_dir() -> Path:
    argv0 = sys.argv[0] if sys.argv and sys.argv[0] else "."
    return Path(argv0).resolve().parent


def _load_yaml(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except FileNotFoundError as e:
        raise ConfigError(f"config file not found: {path}") from e
    except OSError as e:
        raise ConfigError(f"could not read config file {path}: {e}") from e
    except yaml.YAMLError as e:
        raise ConfigError(f"invalid YAML in {path}: {e}") from e

    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigError(
            f"config root must be a mapping, got {type(data).__name__}"
        )
    return data


def _select_parser(
    parsers: dict[str, Callable[..., Any]], version: str
) -> Callable[..., Any]:
    parser = parsers.get(version)
    if parser is None:
        supported = ", ".join(sorted(parsers))
        raise ConfigError(
            f"unsupported config version {version!r}; supported: {supported}"
        )
    return parser


def parse_standalone_config(
    path: str, *, base_path: str | Path | None = None
) -> StandaloneConfigV1:
    base = Path(base_path).resolve() if base_path is not None else _entrypoint_dir()

    with bound_contextvars(path=path):
        log.info("Parsing config file...")
        data = _load_yaml(path)
        version = data.get("version", DEFAULT_CONFIG_VERSION)
        config = _select_parser(_STANDALONE_PARSERS, version)(data, base_path=base)
        log.info("Config file parsed.")
        return config


def parse_service_config(
    path: str, *, base_path: str | Path | None = None
) -> ServiceConfigV1:
    base = Path(base_path).resolve() if base_path is not None else _entrypoint_dir()

    with bound_contextvars(path=path):
        log.info("Parsing config file...")
        data = _load_yaml(path)
        version = data.get("version", DEFAULT_CONFIG_VERSION)
        config = _select_parser(_SERVICE_PARSERS, version)(data, base_path=base)
        log.info("Config file parsed.")
        return config


__all__ = [
    "ConfigError",
    "parse_standalone_config",
    "parse_service_config",
    "StandaloneConfigV1",
    "ServiceConfigV1",
]
