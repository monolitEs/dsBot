"""GigaAM v3 e2e CTC int8, loaded explicitly and retained until close()."""
import ctypes
import gc
from pathlib import Path
import sys

from .transcription import SerializedTranscriber, SttError, read_pcm


class LocalTranscriber(SerializedTranscriber):
    def __init__(self, model_directory: Path):
        super().__init__()
        self._directory = Path(model_directory)
        self._model = None

    def _load(self):
        try:
            import numpy
            import sherpa_onnx
            self._numpy = numpy
            self._model = sherpa_onnx.OfflineRecognizer.from_nemo_ctc(
                model=str(self._directory / 'model.int8.onnx'),
                tokens=str(self._directory / 'tokens.txt'), num_threads=4,
                sample_rate=16000, feature_dim=64, decoding_method='greedy_search',
                provider='cpu')
        except ImportError:
            raise SttError('Install sherpa-onnx and numpy for local STT') from None
        except Exception:
            raise SttError('Local STT model initialization failed') from None

    def _transcribe(self, wav: bytes) -> str:
        pcm = read_pcm(wav)
        try:
            samples = self._numpy.frombuffer(pcm, dtype='<i2').astype(self._numpy.float32) / 32768
            stream = self._model.create_stream()
            stream.accept_waveform(16000, samples)
            self._model.decode_stream(stream)
            return stream.result.text.strip()
        except Exception:
            raise SttError('Local STT recognition failed') from None

    def _unload(self):
        self._model = None
        gc.collect()
        if sys.platform.startswith('linux'):
            try:
                trim = ctypes.CDLL(None).malloc_trim
            except (OSError, AttributeError):
                return
            trim.argtypes = [ctypes.c_size_t]
            trim.restype = ctypes.c_int
            trim(0)
