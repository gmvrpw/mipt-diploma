from PIL.Image import Image


class Character:
    def __init__(self, image: Image):
        self._image = image

    @property
    def image(self):
        return self._image
