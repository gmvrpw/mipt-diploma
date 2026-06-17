# PYTHON_ARGCOMPLETE_OK
import argparse
import logging
import sys
import warnings

import argcomplete
import huggingface_hub
import structlog
from argcomplete.completers import FilesCompleter
from structlog.stdlib import BoundLogger

from src.Service import Service
from src.config import ServiceConfigV1, parse_service_config
from src.infra.loki import LokiProcessor, LokiSink

warnings.filterwarnings('ignore')
huggingface_hub.utils.logging.set_verbosity_error()

log: BoundLogger = structlog.get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='worker-service',
        description='Run worker service in service mode',
    )

    config_arg = parser.add_argument(
        '-c', '--config',
        metavar='CONFIG_FILE',
        required=True,
        help='Path to the service config file',
    )
    config_arg.completer = FilesCompleter()  # type: ignore[attr-defined]

    return parser


def configure_logging(loki_sinks: list[LokiSink] | None = None) -> None:
    processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    if loki_sinks:
        processors.append(LokiProcessor(loki_sinks))
    processors.append(structlog.processors.KeyValueRenderer(
        key_order=["timestamp", "level", "event"],
    ))
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=False,
    )


def main():
    parser = build_parser()
    argcomplete.autocomplete(parser)

    args = parser.parse_args(sys.argv[1:])

    configure_logging()

    log.info("Parsing service config...")

    match parse_service_config(args.config):
        case ServiceConfigV1() as config:
            app = Service(config)

    configure_logging(loki_sinks=app.loki_sinks)

    app.do()


if __name__ == '__main__':
    main()
