import asyncio
import threading
from collections import deque
from typing import Generic, TypeVar, Iterator


T = TypeVar("T")


class PipeClosedError(Exception):
    """Raised on operating with closed pipe."""
    pass


class Package(Generic[T]):
    _data: T | Exception

    def __init__(self, data: T | Exception):
        self._data = data

    def unpack(self) -> T:
        if isinstance(self._data, Exception):
            raise self._data
        return self._data


class Pipe(Generic[T]):
    def __init__(self):
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._queue: deque[Package[T]] = deque()
        self._closed = False

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    def close(self):
        with self._not_empty:
            self._closed = True
            self._not_empty.notify_all()

    def pipe_in(self, value: T | Exception):
        with self._not_empty:
            if self._closed:
                raise PipeClosedError("Cannot write to a closed pipe")
            self._queue.append(Package(value))
            self._not_empty.notify()

    def pipe_out(self, timeout: float | None = None) -> Package[T]:
        with self._not_empty:
            while len(self._queue) == 0:
                if self._closed:
                    raise PipeClosedError("Pipe is closed and empty")
                if not self._not_empty.wait(timeout=timeout):
                    # timeout expired
                    raise TimeoutError("pipe_out timed out")
            return self._queue.popleft()

    async def aclose(self):
        await asyncio.to_thread(self.close)

    async def apipe_in(self, value: T | Exception):
        await asyncio.to_thread(self.pipe_in, value)

    async def apipe_out(self) -> Package[T]:
        return await asyncio.to_thread(self.pipe_out)


class PipeIn(Generic[T]):
    def __init__(self, pipe: Pipe[T]):
        self._pipe = pipe

    @property
    def closed(self):
        return self._pipe.closed

    def pipe(self, value: T | Exception):
        self._pipe.pipe_in(value)

    def close(self):
        self._pipe.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class PipeOut(Generic[T]):
    def __init__(self, pipe: Pipe[T]):
        self._pipe = pipe

    @property
    def closed(self):
        return self._pipe.closed

    def pipe(self, timeout: float | None = None) -> T:
        return self._pipe.pipe_out(timeout=timeout).unpack()

    def close(self):
        self._pipe.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __iter__(self) -> Iterator[Package[T]]:
        return self

    def __next__(self) -> Package[T]:
        try:
            return self._pipe.pipe_out()
        except PipeClosedError:
            raise StopIteration()


class AsyncPipeIn(Generic[T]):
    def __init__(self, pipe: Pipe[T]):
        self._pipe = pipe

    @property
    async def closed(self):
        await asyncio.to_thread(lambda: self._pipe.closed)

    async def pipe(self, value: T | Exception):
        await self._pipe.apipe_in(value)

    async def close(self):
        await self._pipe.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


class AsyncPipeOut(Generic[T]):
    def __init__(self, pipe: Pipe[T]):
        self._pipe = pipe

    @property
    async def closed(self):
        await asyncio.to_thread(lambda: self._pipe.closed)

    async def pipe(self) -> T:
        return (await self._pipe.apipe_out()).unpack()

    async def close(self):
        await self._pipe.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    def __aiter__(self):
        return self

    async def __anext__(self) -> Package[T]:
        try:
            return await self._pipe.apipe_out()
        except PipeClosedError:
            raise StopAsyncIteration()
