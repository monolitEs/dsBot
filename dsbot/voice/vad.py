"""Stateful PCM segmentation for one participant, without a phrase length limit."""
from pathlib import Path
import io
import wave

import numpy as np
from pydub.utils import audioop
import sherpa_onnx

from .transcription import SttError

MODEL_PATH = Path(__file__).resolve().parents[2] / 'models' / 'silero_vad.onnx'


class SpeechDetector:
    def __init__(self):
        if not MODEL_PATH.is_file():
            raise SttError('Missing models/silero_vad.onnx')
        config = sherpa_onnx.VadModelConfig()
        config.silero_vad.model = str(MODEL_PATH)
        config.silero_vad.threshold = 0.5
        config.silero_vad.min_speech_duration = 0.1
        config.silero_vad.min_silence_duration = 0.6
        config.sample_rate = 16000
        config.num_threads = 1
        self.model = sherpa_onnx.VadModel.create(config)
        self.window_bytes = self.model.window_size() * 2
        self.reset()

    def reset(self):
        self.model.reset()
        self.resampler = None
        self.raw = b''
        self.pending = bytearray()
        self.pre_roll = bytearray()
        self.phrase = bytearray()
        self.active = False
        self.last_poll = None

    def feed(self, data: bytes, now: float) -> list[bytes]:
        if data:
            self.raw += data
            count = len(self.raw) // 4 * 4  # PCM16 stereo frames may cross reads.
            mono = audioop.tomono(self.raw[:count], 2, 0.5, 0.5)
            self.raw = self.raw[count:]
            pcm, self.resampler = audioop.ratecv(mono, 2, 1, 48000, 16000, self.resampler)
        elif self.last_poll is not None:
            # Discord may send no packets during silence. Count wall time once.
            pcm = bytes(int(max(0, now - self.last_poll) * 16000) * 2)
        else:
            pcm = b''
        self.last_poll = now
        self.pending.extend(pcm)
        completed = []
        count = len(self.pending) // self.window_bytes * self.window_bytes
        for offset in range(0, count, self.window_bytes):
            frame = self.pending[offset:offset + self.window_bytes]
            samples = np.frombuffer(frame, dtype='<i2').astype(np.float32) / 32768
            speaking = self.model.is_speech(samples)
            if speaking:
                if not self.active:
                    self.phrase.extend(self.pre_roll)
                    self.pre_roll.clear()
                    self.active = True
                    print('STT: Обнаружена речь')
                self.phrase.extend(frame)
            elif self.active:
                self.phrase.extend(frame)
                output = io.BytesIO()
                with wave.open(output, 'wb') as wav:
                    wav.setnchannels(1)
                    wav.setsampwidth(2)
                    wav.setframerate(16000)
                    wav.writeframes(self.phrase)
                completed.append(output.getvalue())
                self.phrase.clear()
                self.active = False
                print(f'VAD: Конец речи, причина={"silence" if data else "no_packets"}')
            else:
                self.pre_roll.extend(frame)
                del self.pre_roll[:-9600]  # Keep 300 ms before confirmed speech.
        del self.pending[:count]
        return completed
