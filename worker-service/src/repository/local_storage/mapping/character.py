import io
from PIL import Image as PILImage

from src.domain.model import Character


def from_character(character: Character) -> bytes:
    buf = io.BytesIO()
    character.image.save(buf, format="PNG")
    return buf.getvalue()


def to_character(b: bytes) -> Character:
    img = PILImage.open(io.BytesIO(b))
    img.load()
    return Character(image=img)
