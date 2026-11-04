"""STT contract for complete mono PCM16/16kHz WAV recordings."""
from abc import ABC, abstractmethod
import asyncio
import io
from typing import Protocol
import wave


class SttError(RuntimeError):
    """A controlled STT failure safe to show without request internals."""


class QuotaExceeded(SttError):
    """The cloud provider confirmed its daily quota was exhausted."""


class Transcriber(Protocol):
    async def start(self) -> None: ...
    async def transcribe(self, wav: bytes) -> str: ...
    async def close(self) -> None: ...


def read_pcm(wav: bytes) -> bytes:
    try:
        with wave.open(io.BytesIO(wav), 'rb') as audio:
            if (audio.getnchannels(), audio.getsampwidth(), audio.getframerate(),
                    audio.getcomptype()) != (1, 2, 16000, 'NONE'):
                raise SttError('Expected mono PCM16/16000Hz WAV')
            frames = audio.getnframes()
            pcm = audio.readframes(frames)
            if not frames or len(pcm) != frames * 2:
                raise SttError('Empty or truncated WAV')
            return pcm
    except (wave.Error, EOFError):
        raise SttError('Invalid WAV') from None


async def run_blocking(operation, *args):
    # Cancellation of an await does not stop its native thread. Drain it before
    # releasing the caller's lock, even if cancellation is requested repeatedly.
    task = asyncio.create_task(asyncio.to_thread(operation, *args))
    cancelled = False
    while True:
        try:
            result = await asyncio.shield(task)
            break
        except asyncio.CancelledError:
            if task.cancelled():  # Event-loop shutdown can cancel the internal task too.
                raise
            cancelled = True
        except Exception:
            if cancelled:
                raise asyncio.CancelledError from None
            raise
    if cancelled:
        raise asyncio.CancelledError
    return result


class SerializedTranscriber(ABC):
    """Share start/close serialization and cancellation handling across backends."""

    def __init__(self):
        self._lock = asyncio.Lock()
        self._started = False

    async def start(self) -> None:
        async with self._lock:
            if not self._started:
                try:
                    await run_blocking(self._load)
                except asyncio.CancelledError:
                    await run_blocking(self._unload)
                    raise
                self._started = True

    async def transcribe(self, wav: bytes) -> str:
        async with self._lock:
            if not self._started:
                raise SttError('STT backend is not started')
            return await run_blocking(self._transcribe, wav)

    async def close(self) -> None:
        async with self._lock:
            if self._started:
                self._started = False
                await run_blocking(self._unload)

    def _load(self):
        # Backends without a persistent model need no loading step.
        pass

    def _unload(self):
        # Backends without persistent resources need no unloading step.
        pass

    @abstractmethod
    def _transcribe(self, wav: bytes) -> str:
        pass
