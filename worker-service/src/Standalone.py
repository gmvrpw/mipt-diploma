import signal
import threading

import structlog
from structlog.stdlib import BoundLogger

from src.config.standalone_v1 import StandaloneConfigV1

from src.app.ticket import TicketService
from src.app.add import AddService
from src.app.animate import AnimateService
from src.app.delete import DeleteService
from src.app.norm_map import NormMapService
from src.app.rmbg import RmbgService
from src.app.stand import StandService
from src.controller.task.local import LocalTaskController
from src.domain.worker.scheduler.Scheduler import Scheduler
from src.domain.worker.device.eviction.LruEvictionPolicy import LruEvictionPolicy
from src.domain.worker.device.Device import Device
from src.domain.worker.Worker import Worker
from src.domain.pipeline import (
    InspyreNetRmbgPipeline,
    QwenImageEditAddPipeline,
    QwenImageEditDeletePipeline,
    QwenImageEditStandPipeline,
    StableNormal01NormalMapPipeline,
    Wan22AnimatePipeline,
)
from src.domain.pipeline.model.InSPyReNetBackboneModel import InSPyReNetBackboneModel
from src.domain.pipeline.model.InSPyReNetDecoderModel import InSPyReNetDecoderModel
from src.domain.pipeline.model.StableNormalModel import StableNormalModel
from src.repository.local_storage.repository import LocalStorageRepository
from src.infra.structlog.standalone import END_SECTION
from src.IO import IO


log: BoundLogger = structlog.get_logger(__name__)


class Standalone:
    def __init__(self, config: StandaloneConfigV1):
        self._version = config.version

        scheduler = Scheduler()
        device = Device(
            eviction_policy=LruEvictionPolicy(),
            gpu_mem=int(config.device.gpu_mem),
        )

        repository = LocalStorageRepository(
            base_path=config.outputs.dir,
            read_timeout=config.outputs.timeout.read,
            read_retries=config.outputs.retries.read,
            write_timeout=config.outputs.timeout.write,
            write_retries=config.outputs.retries.write,
        )

        ticket = TicketService(get_ticket=scheduler.queue)

        inspyre_backbone = InSPyReNetBackboneModel(
            device=device, model_dir=config.models.dir)
        inspyre_decoder = InSPyReNetDecoderModel(
            device=device, model_dir=config.models.dir)
        stable_normal = StableNormalModel(
            device=device, model_dir=config.models.dir)

        self._qwen_image_edit_add = QwenImageEditAddPipeline(
            device=device,
            model_dir=config.models.dir,
            hf_path=config.capabilities.add.hf_path,
        )
        self._qwen_image_edit_delete = QwenImageEditDeletePipeline(
            device=device,
            model_dir=config.models.dir,
            hf_path=config.capabilities.delete.hf_path,
        )
        self._qwen_image_edit_stand = QwenImageEditStandPipeline(
            device=device,
            model_dir=config.models.dir,
            hf_path=config.capabilities.stand.hf_path,
        )
        self._wan22_animate = Wan22AnimatePipeline(
            device=device,
            model_dir=config.models.dir,
            hf_path=config.capabilities.animate.hf_path,
        )
        self._inspyre_net_rmbg = InspyreNetRmbgPipeline(
            device=device,
            backbone=inspyre_backbone,
            decoder=inspyre_decoder,
            batch_size=4,
            hf_path=config.capabilities.rmbg.hf_path,
        )
        self._stable_normal_normal_map = StableNormal01NormalMapPipeline(
            device=device,
            model=stable_normal,
            batch_size=4,
            hf_path=config.capabilities.norm_map.hf_path,
        )

        add = AddService(
            get_character=repository.get_character,
            save_character=repository.save_character,
            pipeline=self._qwen_image_edit_add,
        )
        delete = DeleteService(
            get_character=repository.get_character,
            save_character=repository.save_character,
            pipeline=self._qwen_image_edit_delete,
        )
        stand = StandService(
            get_character=repository.get_character,
            save_character=repository.save_character,
            pipeline=self._qwen_image_edit_stand,
        )
        animate = AnimateService(
            get_character=repository.get_character,
            save_frames=repository.save_frames,
            pipeline=self._wan22_animate,
        )
        rmbg = RmbgService(
            get_frames=repository.get_frames,
            save_frames=repository.save_frames,
            pipeline=self._inspyre_net_rmbg,
        )
        norm_map = NormMapService(
            get_frames=repository.get_frames,
            save_frames=repository.save_frames,
            pipeline=self._stable_normal_normal_map,
        )

        self._controller = LocalTaskController(
            ticket=ticket,
            add=add,
            delete=delete,
            stand=stand,
            animate=animate,
            rmbg=rmbg,
            norm_map=norm_map,
        )

        self._io = IO()
        self._worker = Worker(scheduler=scheduler, device=device)

    def do(self, path: str, inputs: dict[str, str]):
        log.info("Service starting...",
                 task_name="Startup Service", version=self._version)

        done = threading.Event()

        signal.signal(signal.SIGTERM, lambda *_: done.set())
        signal.signal(signal.SIGINT, lambda *_: done.set())

        def run_worker():
            try:
                self._worker.start()
            except Exception:
                log.exception("Worker failed")
            finally:
                done.set()

        worker_thread = threading.Thread(target=run_worker)
        worker_thread.start()

        def run_io():
            try:
                self._io.start([self._controller.run(path, inputs)])
            except Exception:
                log.exception("IO failed")
            finally:
                done.set()

        io_thread = threading.Thread(target=run_io)
        io_thread.start()

        log.info("Service started.")

        done.wait()

        log.info("Shutting down...", task_name="Shutdown Service")

        self._io.stop()
        io_thread.join()

        self._worker.stop()
        worker_thread.join()

        log.info("Service stopped.", task_name=END_SECTION)
