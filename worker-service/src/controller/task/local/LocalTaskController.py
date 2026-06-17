import structlog
from structlog.stdlib import BoundLogger

from src.app import (
    AddService,
    AnimateService,
    DeleteService,
    NormMapService,
    RmbgService,
    StandService,
    TicketService,
)

from .pipeline import parse_pipeline, topsort
from .pipeline.tasks import (
    AddTask,
    AnimateTask,
    DeleteTask,
    NormMapTask,
    RmbgTask,
    StandTask,
)

from .mapping import (
    from_add_response, to_add_request,
    from_animate_response, to_animate_request,
    from_delete_response, to_delete_request,
    from_norm_map_response, to_norm_map_request,
    from_rmbg_response, to_rmbg_request,
    from_stand_response, to_stand_request,
)

log: BoundLogger = structlog.get_logger(__name__)


class LocalTaskController:
    def __init__(
        self,
        ticket: TicketService,
        add: AddService,
        delete: DeleteService,
        stand: StandService,
        animate: AnimateService,
        rmbg: RmbgService,
        norm_map: NormMapService,
    ):
        self._ticket = ticket
        self._add = add
        self._delete = delete
        self._stand = stand
        self._animate = animate
        self._rmbg = rmbg
        self._norm_map = norm_map

    async def run(self, path: str, inputs: dict[str, str]):
        log.info("Parsing pipeline config...",
                 task_name="Parse Pipeline Config", pipeline_path=path)

        try:
            pipeline = parse_pipeline(path, inputs)
            tasks = topsort(pipeline)
        except ValueError:
            log.exception("Failed to parse pipeline file.")
            return

        resources = {f"pipeline.inputs.{k}": v
                     for k, v in pipeline.inputs.items()}

        for task in tasks:
            log.info("Processing task...",
                     task_name=f"{pipeline.name} / {task.name}")

            ticket = await self._ticket()

            try:
                match task:
                    case AddTask():
                        artifacts = from_add_response(
                            await self._add(to_add_request(task, resources), ticket))
                    case DeleteTask():
                        artifacts = from_delete_response(
                            await self._delete(to_delete_request(task, resources), ticket))
                    case StandTask():
                        artifacts = from_stand_response(
                            await self._stand(to_stand_request(task, resources), ticket))
                    case AnimateTask():
                        artifacts = from_animate_response(
                            await self._animate(to_animate_request(task, resources), ticket))
                    case RmbgTask():
                        artifacts = from_rmbg_response(
                            await self._rmbg(to_rmbg_request(task, resources), ticket))
                    case NormMapTask():
                        artifacts = from_norm_map_response(
                            await self._norm_map(to_norm_map_request(task, resources), ticket))
                    case _:
                        raise Exception()
                resources.update(artifacts)
            except Exception:
                log.exception("Failed to process task.")
                return

            log.info("Task completed.", artifacts=artifacts)

        log.info("Pipeline completed.", task_name="Run Pipeline")
