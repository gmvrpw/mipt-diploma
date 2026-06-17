from dataclasses import dataclass
from typing import overload

from PIL import Image as PILImage
from PIL.Image import Image


@dataclass
class Frames:
    atlas: Image
    width: int
    height: int
    count: int
    offset: int = 0

    def __len__(self) -> int:
        return self.count

    @overload
    def __getitem__(self, key: int) -> Image: ...

    @overload
    def __getitem__(self, key: slice) -> "Frames": ...

    def __getitem__(self, key):
        if isinstance(key, int):
            if key < 0:
                key += self.count
            if key < 0 or key >= self.count:
                raise IndexError("frame index out of range")
            real_index = self.offset + key
            cols = self.atlas.width // self.width
            row = real_index // cols
            col = real_index % cols
            box = (
                col * self.width,
                row * self.height,
                (col + 1) * self.width,
                (row + 1) * self.height,
            )
            return self.atlas.crop(box)

        if isinstance(key, slice):
            indices = list(range(*key.indices(self.count)))
            if not indices:
                return Frames(
                    atlas=self.atlas,
                    width=self.width,
                    height=self.height,
                    count=0,
                    offset=self.offset,
                )

            # Если slice непрерывный с шагом 1 — просто меняем offset/count
            step = key.step if key.step is not None else 1
            if step == 1:
                return Frames(
                    atlas=self.atlas,
                    width=self.width,
                    height=self.height,
                    count=len(indices),
                    offset=self.offset + indices[0],
                )

            # Иначе перестраиваем атлас из выбранных кадров
            new_count = len(indices)
            new_atlas = PILImage.new(
                self.atlas.mode, (self.width * new_count, self.height)
            )
            for i, idx in enumerate(indices):
                frame = self[idx]
                new_atlas.paste(frame, (i * self.width, 0))
            return Frames(
                atlas=new_atlas,
                width=self.width,
                height=self.height,
                count=new_count,
                offset=0,
            )

        raise TypeError(f"Invalid key type: {type(key)}")
