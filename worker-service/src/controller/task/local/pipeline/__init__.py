import structlog
from structlog.contextvars import bound_contextvars
from structlog.stdlib import BoundLogger

import yaml

from .pipeline import Pipeline
from .v1 import parse as parse_v1

from .topsort import topsort


log: BoundLogger = structlog.get_logger(__name__)


def parse_pipeline(path: str, inputs: dict[str, str]) -> Pipeline:
    with bound_contextvars(path=path):
        log.info("Parsing pipeline file...")

        with open(path) as f:
            data = yaml.safe_load(f)

        version = data.get("version")

        match version:
            case "v1":
                pipeline = parse_v1(data, inputs)
            case _:
                raise ValueError(
                    f"Unsupported pipeline version: '{version}'")

        log.info("Pipeline file parsed.")
        return pipeline


__all__ = ["parse_pipeline", "topsort", "Pipeline"]
