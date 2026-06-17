# PYTHON_ARGCOMPLETE_OK
from diffusers.utils import logging as df_logging
from transformers.utils import logging as tf_logging
import warnings
import argparse
import logging
import re
import os
import sys
import argcomplete
import structlog
from argcomplete.completers import FilesCompleter
from structlog.stdlib import BoundLogger

from src.Standalone import Standalone
from src.config import StandaloneConfigV1, parse_standalone_config
from src.infra.structlog.standalone import TerminalRenderer

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["DIFFUSERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

warnings.filterwarnings("ignore")

tf_logging.set_verbosity_error()
tf_logging.disable_progress_bar()

df_logging.set_verbosity_error()
df_logging.disable_progress_bar()

_INPUTS_RE = re.compile(r'^--inputs\.([^=]+)=(.+)$')

log: BoundLogger = structlog.get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='worker-standalone',
        description='Run worker service as standalone application',
    )

    config_arg = parser.add_argument(
        '-c', '--config',
        metavar='CONFIG_FILE',
        help='Path to the service config file',
    )
    config_arg.completer = FilesCompleter()  # type: ignore[attr-defined]

    pipeline_arg = parser.add_argument(
        'pipeline',
        metavar='PIPELINE_FILE',
        help='Path to pipeline config file',
    )
    pipeline_arg.completer = FilesCompleter()  # type: ignore[attr-defined]

    return parser


def parse_args(parser: argparse.ArgumentParser, argv: list[str]):
    return parser.parse_args(argv)


def parse_inputs(argv: list[str]) -> dict[str, str]:
    inputs: dict[str, str] = {}

    for arg in argv:
        if m := _INPUTS_RE.match(arg):
            input_id, input_data = m.group(1), m.group(2)
            inputs[input_id] = input_data

    return inputs


def configure_logging(renderer: TerminalRenderer) -> None:
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="%H:%M:%S", utc=False),
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def main():
    parser = build_parser()
    argcomplete.autocomplete(parser)

    argv = sys.argv[1:]

    sep = argv.index('--') if '--' in argv else len(argv)

    args = parse_args(parser, argv[:sep])
    inputs = parse_inputs(argv[sep+1:])

    configure_logging(TerminalRenderer())

    log.info("Parsing service config...", task_name="Parse Service Config")

    match parse_standalone_config(args.config):
        case StandaloneConfigV1() as config:
            app = Standalone(config)

    app.do(args.pipeline, inputs)


if __name__ == '__main__':
    main()
