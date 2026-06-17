import io
import json

from PIL import Image as PILImage

from src.domain.model import Frames


def from_frames(frames: Frames) -> tuple[bytes, bytes]:
    cols = frames.atlas.width // frames.width
    total_in_atlas = (frames.atlas.width // frames.width) * (
        frames.atlas.height // frames.height
    )

    if frames.offset == 0 and frames.count == total_in_atlas:
        atlas_to_save = frames.atlas
    else:
        atlas_to_save = PILImage.new(
            frames.atlas.mode, (frames.width * frames.count, frames.height)
        )
        for i in range(frames.count):
            real_index = frames.offset + i
            row = real_index // cols
            col = real_index % cols
            box = (
                col * frames.width,
                row * frames.height,
                (col + 1) * frames.width,
                (row + 1) * frames.height,
            )
            atlas_to_save.paste(frames.atlas.crop(box), (i * frames.width, 0))

    buf = io.BytesIO()
    atlas_to_save.save(buf, format="PNG")
    atlas_bytes = buf.getvalue()

    meta = {
        "width": frames.width,
        "height": frames.height,
        "count": frames.count,
    }
    meta_bytes = json.dumps(meta).encode("utf-8")
    return atlas_bytes, meta_bytes


def to_frames(atlas_bytes: bytes, meta_bytes: bytes | None) -> Frames:
    atlas = PILImage.open(io.BytesIO(atlas_bytes))
    atlas.load()

    if meta_bytes is None:
        h = atlas.height
        return Frames(
            atlas=atlas,
            width=h,
            height=h,
            count=atlas.width // h,
            offset=0,
        )

    meta = json.loads(meta_bytes.decode("utf-8"))
    return Frames(
        atlas=atlas,
        width=meta["width"],
        height=meta["height"],
        count=meta["count"],
        offset=0,
    )
