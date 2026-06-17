import random
from itertools import cycle
from typing import Callable, Iterator

LoaderFactory = Callable[[], Iterator[str]]


def blocks() -> Iterator[str]:
    return cycle("▁▂▃▄▅▆▇█▇▆▅▄▃▂")


def braille_spin() -> Iterator[str]:
    return cycle("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏")


def braille_bounce() -> Iterator[str]:
    return cycle("⣾⣽⣻⢿⡿⣟⣯⣷")


def arrows() -> Iterator[str]:
    return cycle("←↖↑↗→↘↓↙")


def moon() -> Iterator[str]:
    return cycle("◐◓◑◒")


def triangles() -> Iterator[str]:
    return cycle("◢◣◤◥")


def square_corners() -> Iterator[str]:
    return cycle("◰◱◲◳")


_ALL: tuple[LoaderFactory, ...] = (
    blocks,
    braille_spin,
    braille_bounce,
    arrows,
    moon,
    triangles,
    square_corners,
)


def get_random_loader() -> Iterator[str]:
    """Return a fresh iterator from a randomly chosen loader."""
    return random.choice(_ALL)()


__all__ = [
    "LoaderFactory",
    "blocks",
    "braille_spin",
    "braille_bounce",
    "arrows",
    "moon",
    "triangles",
    "square_corners",
    "get_random_loader",
]

