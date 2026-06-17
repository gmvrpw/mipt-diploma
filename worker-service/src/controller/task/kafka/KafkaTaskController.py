import time
from asyncio import CancelledError, Task, get_event_loop

import betterproto2
import worker_service_proto.task as proto

import structlog
from structlog.contextvars import bound_contextvars
from structlog.stdlib import BoundLogger

from src import metrics
from src.app import (
    AddService,
    AnimateService,
    CancellationService,
    DeleteService,
    NormMapService,
    RmbgService,
    StandService,
    TicketService,
)
from src.app.cancellation import was_task_cancelled

from src.domain.model.error import DomainError
from src.domain.worker import Ticket

from src.infra.kafka import KafkaMessage, KafkaController

from .config import KafkaControllerConfig
from .mapping import (
    from_add_response,
    from_animate_response,
    from_delete_response,
    from_norm_map_response,
    from_rmbg_response,
    from_stand_response,
    to_add_request,
    to_animate_request,
    to_cancelled_response,
    to_delete_request,
    to_norm_map_request,
    to_rmbg_request,
    to_stand_request,
)
from .mapping.failed import domain_error_to_code, to_failed_response

log: BoundLogger = structlog.get_logger(__name__)


class KafkaTaskController(KafkaController):
    def __init__(
        self,
        config: KafkaControllerConfig,
        cancel: CancellationService,
        ticket: TicketService,
        add: AddService,
        delete: DeleteService,
        stand: StandService,
        animate: AnimateService,
        rmbg: RmbgService,
        norm_map: NormMapService,
    ):
        super().__init__(
            bootstrap_servers=config.bootstrap_servers,
            client_id=config.client_id,
            group_id=config.group_id,
            consume_topics=config.topics,
            produce_topic=config.produce_topic,
        )

        self._cancel = cancel
        self._ticket = ticket
        self._add = add
        self._delete = delete
        self._stand = stand
        self._animate = animate
        self._rmbg = rmbg
        self._norm_map = norm_map

    async def handle(self, message: KafkaMessage):
        try:
            request = proto.Task.parse(message.value)
            [task_type, task] = betterproto2.which_one_of(request, "task")

            if task is None:
                raise ValueError("Message does not contain any task")

            metrics.tasks_received_total.labels(task_type=task_type).inc()

            ticket = await self._ticket()

            with bound_contextvars(task_id=task.id):
                match task_type:
                    case "add":
                        proc = self._cancel.create_cancellable_task(
                            task.id, self._handle_add(message, task, ticket))
                    case "delete":
                        proc = self._cancel.create_cancellable_task(
                            task.id, self._handle_delete(message, task, ticket))
                    case "stand":
                        proc = self._cancel.create_cancellable_task(
                            task.id, self._handle_stand(message, task, ticket))
                    case "animate":
                        proc = self._cancel.create_cancellable_task(
                            task.id, self._handle_animate(message, task, ticket))
                    case "rmbg":
                        proc = self._cancel.create_cancellable_task(
                            task.id, self._handle_rmbg(message, task, ticket))
                    case "norm_map":
                        proc = self._cancel.create_cancellable_task(
                            task.id, self._handle_norm_map(message, task, ticket))
                    case _:
                        raise ValueError(f"Unknown task type '{task_type}'")

                get_event_loop().create_task(
                    self._handle(message, proc, task_type, task.id))
        except ValueError as e:
            log.warning(f"Message skipped: {e.args[0]}",
                        topic=message.topic, partition=message.partition, offset=message.offset)
            await self.commit(message)

    async def _handle(
        self,
        message: KafkaMessage,
        task: Task,
        task_type: str,
        task_id: str,
    ):
        started = time.monotonic()
        try:
            await task
            metrics.tasks_completed_total.labels(task_type=task_type).inc()
        except CancelledError:
            if was_task_cancelled(task):
                metrics.tasks_cancelled_total.labels(task_type=task_type).inc()
                await self._task_cancelled(message, task_id)
            else:
                metrics.tasks_failed_total.labels(task_type=task_type).inc()
        except Exception as e:
            metrics.tasks_failed_total.labels(task_type=task_type).inc()
            await self._handle_exception(message, e, task_id)
        finally:
            metrics.ticket_duration_seconds.labels(
                task_type=task_type).observe(time.monotonic() - started)

    async def _handle_add(self, msg: KafkaMessage, task: proto.Add, ticket: Ticket):
        await self.response(msg, bytes(from_add_response(
            id=task.id,
            response=await self._add(to_add_request(task), ticket),
        )))

    async def _handle_delete(self, msg: KafkaMessage, task: proto.Delete, ticket: Ticket):
        await self.response(msg, bytes(from_delete_response(
            id=task.id,
            response=await self._delete(to_delete_request(task), ticket),
        )))

    async def _handle_stand(self, msg: KafkaMessage, task: proto.Stand, ticket: Ticket):
        await self.response(msg, bytes(from_stand_response(
            id=task.id,
            response=await self._stand(to_stand_request(task), ticket),
        )))

    async def _handle_animate(self, msg: KafkaMessage, task: proto.Animate, ticket: Ticket):
        await self.response(msg, bytes(from_animate_response(
            id=task.id,
            response=await self._animate(to_animate_request(task), ticket),
        )))

    async def _handle_rmbg(self, msg: KafkaMessage, task: proto.Rmbg, ticket: Ticket):
        await self.response(msg, bytes(from_rmbg_response(
            id=task.id,
            response=await self._rmbg(to_rmbg_request(task), ticket),
        )))

    async def _handle_norm_map(self, msg: KafkaMessage, task: proto.NormMap, ticket: Ticket):
        await self.response(msg, bytes(from_norm_map_response(
            id=task.id,
            response=await self._norm_map(to_norm_map_request(task), ticket),
        )))

    async def _task_cancelled(self, msg: KafkaMessage, task_id: str):
        await self.response(msg, bytes(to_cancelled_response(task_id)))

    async def _handle_exception(
        self,
        msg: KafkaMessage,
        e: BaseException,
        task_id: str,
    ):
        if isinstance(e, DomainError):
            code = domain_error_to_code(e)
            message = e.args[0] if e.args else type(e).__name__
            log.warning("Task failed with domain error",
                        error_code=code, error_message=str(message))
        else:
            code = "internal"
            message = f"{type(e).__name__}: {e}" if str(e) else type(e).__name__
            log.exception("Task failed with unhandled exception")

        await self.response(
            msg,
            bytes(to_failed_response(task_id, code, str(message))),
        )
