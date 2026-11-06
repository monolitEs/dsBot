"""Finite gTTS responses played incrementally as Discord PCM frames."""
import queue
import subprocess
import threading
import time

import discord
from gtts import gTTS


class StreamingTTS(discord.AudioSource):
    FRAME_SIZE = 3840  # 20 ms, stereo PCM16 at 48 kHz.
    SILENCE = bytes(FRAME_SIZE)

    def __init__(self, text):
        self._text = text
        self._frames = queue.Queue(maxsize=50)
        self._stopped = threading.Event()
        self._done = threading.Event()
        self._lock = threading.Lock()
        self._decoder = None
        self._started = False
        self._first_frame = True
        self._start_time = time.monotonic()
        self._worker = threading.Thread(target=self._produce, daemon=True,
                                        name='tts-producer')

    def _put(self, frame):
        while not self._stopped.is_set():
            try:
                self._frames.put(frame, timeout=0.05)
                return True
            except queue.Full:
                continue
        return False

    def _decode(self, mp3):
        with self._lock:
            if self._stopped.is_set():
                return b''
            decoder = subprocess.Popen(
                ['ffmpeg', '-loglevel', 'error', '-f', 'mp3', '-i', 'pipe:0',
                 '-f', 's16le', '-ar', '48000', '-ac', '2', 'pipe:1'],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            self._decoder = decoder
        try:
            with decoder:
                pcm, _ = decoder.communicate(mp3)
                if decoder.returncode or not pcm:
                    raise RuntimeError('FFmpeg could not decode TTS audio')
                return pcm
        finally:
            with self._lock:
                self._decoder = None

    def _queue_pcm(self, pcm):
        end = len(pcm) - len(pcm) % self.FRAME_SIZE
        for offset in range(0, end, self.FRAME_SIZE):
            if not self._put(pcm[offset:offset + self.FRAME_SIZE]):
                return None
        return pcm[end:]

    def _produce(self):
        chunks = None
        try:
            chunks = gTTS(self._text, lang='ru', timeout=(5, 15)).stream()
            pending = b''
            while not self._stopped.is_set():
                try:
                    mp3 = next(chunks)
                except StopIteration:
                    break
                if self._stopped.is_set():
                    return
                pending = self._queue_pcm(pending + self._decode(mp3))
                if pending is None:
                    return
            if pending and not self._stopped.is_set():
                self._put(pending.ljust(self.FRAME_SIZE, b'\0'))
            if not self._stopped.is_set():
                elapsed = time.monotonic() - self._start_time
                print(f'TTS: Загрузка и подготовка всей речи: {elapsed:.3f} с')
        except Exception as error:
            if not self._stopped.is_set():
                print(f'TTS: Ошибка озвучки: {type(error).__name__}')
        finally:
            if chunks is not None:
                chunks.close()
            self._done.set()

    def read(self):
        with self._lock:
            if self._stopped.is_set():
                return b''
            if not self._started:
                self._started = True
                self._worker.start()
        try:
            frame = self._frames.get_nowait()
        except queue.Empty:
            if not self._done.is_set():
                return self.SILENCE
            # A producer can enqueue its final frame between get() and done.set().
            try:
                frame = self._frames.get_nowait()
            except queue.Empty:
                return b''
        if self._first_frame:
            self._first_frame = False
            elapsed = time.monotonic() - self._start_time
            print(f'TTS: Первый фрагмент передан плееру: {elapsed:.3f} с')
        return frame

    def cleanup(self):
        with self._lock:
            self._stopped.set()
            if self._decoder is not None and self._decoder.poll() is None:
                try:
                    self._decoder.kill()
                except OSError:
                    pass  # The decoder can exit between poll() and kill().
