from dataclasses import dataclass
from functools import partial
from typing import Protocol, cast

from PIL import Image

from src.domain.Pipe import AsyncPipeOut, Pipe, PipeIn
from src.domain.model import Character
from src.domain.model.error import InferenceCanceledError
from src.domain.pipeline._qwen_image_edit_base_2 import QwenImageEditPlusPipeline
from src.domain.worker import Ticket
from src.domain.worker.device.Device import Device


@dataclass
class StandPipelineOutput:
    character: Character


class StandPipeline(Protocol):
    def pipe(
        self,
        character: Character,
        pose: Character | None,
        prompt: str,
        negative_prompt: str | None,
        ticket: Ticket,
    ) -> AsyncPipeOut[StandPipelineOutput]: ...


class QwenImageEditStandPipeline:
    def __init__(
        self, device: Device, model_dir: str, hf_path: str,
    ) -> None:
        self._qwen = QwenImageEditPlusPipeline.shared(
            device=device, model_dir=model_dir, hf_path=hf_path,
        )

    def pipe(
        self,
        character: Character,
        pose: Character | None,
        prompt: str,
        negative_prompt: str | None,
        ticket: Ticket,
    ) -> AsyncPipeOut[StandPipelineOutput]:
        pipe = Pipe[StandPipelineOutput]()
        inp, out = PipeIn(pipe), AsyncPipeOut(pipe)
        ticket.use(partial(self._run,
                           character, pose, prompt, negative_prompt, inp))
        return out

    def _run(
        self,
        character: Character,
        pose: Character | None,
        prompt: str,
        negative_prompt: str | None,
        output: PipeIn[StandPipelineOutput],
    ) -> None:
        try:
            image = [character.image]
            if pose is not None:
                image.append(pose.image)

            result = self._qwen(
                image=image,
                prompt=prompt,
                negative_prompt=negative_prompt,
                on_step=_cancel_on_closed(output),
            )
            images = cast(list[Image.Image], result)
            output.pipe(StandPipelineOutput(character=Character(images[0])))
            output.close()
        except InferenceCanceledError:
            return


def _cancel_on_closed(output: PipeIn):
    def check() -> None:
        if output.closed:
            raise InferenceCanceledError()
    return check
