from pydub import AudioSegment
import io
import asyncio
from .vad import SpeechDetector
from .transcription import QuotaExceeded, SttError, Transcriber, run_blocking


class VoiceSession:
    """Models and quota state live only for one guild voice connection."""

    def __init__(self, cloud: Transcriber, local: Transcriber):
        self.cloud, self.local = cloud, local
        self.detector = self.listener = None
        self.closed = self.fallback = False
        self.lock, self.record_lock = asyncio.Lock(), asyncio.Lock()

    async def start(self):
        async with self.lock:
            if self.closed:
                raise SttError('Voice session is closed')
            if self.detector is None:
                try:
                    await self.cloud.start()
                    await self.local.start()
                    self.detector = await run_blocking(SpeechDetector)
                    print('STT: Модели распознавания речи загружены, бот готов слушать')
                except BaseException:
                    await self.local.close()
                    await self.cloud.close()
                    raise

    async def detect(self, pcm, now):
        async with self.lock:
            if self.closed:
                return []
            return await run_blocking(self.detector.feed, pcm, now)

    async def reset_detector(self):
        async with self.lock:
            if self.detector is not None:
                await run_blocking(self.detector.reset)

    async def transcribe(self, filename: str) -> str:
        async with self.lock:
            if self.closed:
                raise SttError('Voice session is closed')
            wav = await run_blocking(lambda: AudioSegment.from_wav(filename).set_frame_rate(16000)
                                     .export(io.BytesIO(), format='wav').getvalue())
            if not self.fallback:
                try:
                    return await self.cloud.transcribe(wav)
                except QuotaExceeded:
                    self.fallback = True
                    print('STT backend=local reason=quota')
            return await self.local.transcribe(wav)

    async def stop(self):
        if self.listener:
            listener, self.listener = self.listener, None
            listener.cancel()
            if listener.get_task():
                await asyncio.gather(listener.get_task(), return_exceptions=True)

    async def close(self):
        self.closed = True
        async with self.record_lock:
            await self.stop()
            async with self.lock:
                self.detector = None
                await self.local.close()
                await self.cloud.close()
