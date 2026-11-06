"""Streaming segmentation contracts, plus the actual repository ONNX model."""
from contextlib import redirect_stdout
import io
import itertools
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import numpy as np
from pydub.utils import audioop

from voice_interface.transcription import SttError, read_pcm
from voice_interface.vad import SpeechDetector


def stereo(samples):
    return np.repeat(np.repeat(samples, 3), 2).astype('<i2').tobytes()


class VadTests(unittest.TestCase):
    def setUp(self):
        self.output = io.StringIO()
        self.logs = redirect_stdout(self.output)
        self.logs.__enter__()
        self.addCleanup(self.logs.__exit__, None, None, None)

    def detector(self, states):
        model = Mock()
        model.window_size.return_value = 512
        model.is_speech.side_effect = states
        with patch('sherpa_onnx.VadModel.create', return_value=model) as create:
            detector = SpeechDetector()
        config = create.call_args.args[0]
        self.assertAlmostEqual(config.silero_vad.min_silence_duration, 0.6)
        self.assertAlmostEqual(config.silero_vad.min_speech_duration, 0.1)
        return detector

    def test_fragmented_pcm_resamples_and_segments_identically_to_whole_input(self):
        source = (np.arange(512 * 30) % 20000).astype('<i2')
        data = stereo(source)
        states = [False] * 12 + [True] * 5 + [False] * 13
        whole = self.detector(iter(states)).feed(data, 0)
        fragmented = self.detector(iter(states))
        recordings = []
        sizes = itertools.cycle([1, 3, 777, 6144, 2000])
        offset = 0
        while offset < len(data):
            size = next(sizes)
            recordings.extend(fragmented.feed(data[offset:offset + size], 0))
            offset += size
        self.assertEqual(recordings, whole)
        expected = source.tobytes()[12 * 1024 - 9600:18 * 1024]
        self.assertEqual(read_pcm(whole[0]), expected)
        self.assertFalse(fragmented.active)
        self.assertEqual(fragmented.phrase, b'')
        self.assertLess(len(fragmented.pending), 1024)

    def test_no_packet_time_counts_once_and_completes_without_next_packet(self):
        detector = self.detector(iter([True] * 19 + [False] * 100))
        self.assertEqual(detector.feed(stereo(np.ones(512, dtype='<i2')), 10), [])
        completed = []
        for tick in range(1, 15):
            completed.extend(detector.feed(b'', 10 + tick * 0.05))
        self.assertEqual(len(completed), 1)
        self.assertIn('no_packets', self.output.getvalue())
        self.assertEqual(detector.feed(b'', 10.7), [])
        self.assertLessEqual(len(read_pcm(completed[0])), 20 * 1024)

    def test_long_phrase_is_not_cut_at_library_default_duration(self):
        detector = self.detector(itertools.chain(itertools.repeat(True, 3000), [False]))
        self.assertEqual(detector.feed(stereo(np.ones(512 * 3000, dtype='<i2')), 0), [])
        completed = detector.feed(stereo(np.zeros(512, dtype='<i2')), 96.032)
        self.assertEqual(len(read_pcm(completed[0])), 3001 * 1024)
        self.assertEqual(detector.phrase, b'')

    def test_short_pause_and_resumed_speech_remain_one_phrase(self):
        # The model's smoothed speech decision stays true over a 320 ms pause.
        states = [True] * 39 + [False]
        detector = self.detector(iter(states))
        first = np.ones(512 * 5, dtype='<i2')
        pause = np.zeros(512 * 10, dtype='<i2')
        second = np.full(512 * 5, 2, dtype='<i2')
        tail = np.zeros(512 * 20, dtype='<i2')
        self.assertEqual(detector.feed(stereo(first), 0.16), [])
        self.assertEqual(detector.feed(stereo(pause), 0.48), [])
        self.assertEqual(detector.feed(stereo(second), 0.64), [])
        completed = detector.feed(stereo(tail), 1.28)
        self.assertEqual(len(completed), 1)
        expected = np.concatenate([first, pause, second, tail]).tobytes()
        self.assertEqual(read_pcm(completed[0]), expected)

    def test_reset_discards_partial_phrase_and_resampling_state(self):
        detector = self.detector(itertools.repeat(True))
        detector.feed(stereo(np.ones(1024, dtype='<i2')) + b'x', 0)
        self.assertTrue(detector.active)
        detector.reset()
        self.assertFalse(detector.active)
        self.assertEqual(detector.raw, b'')
        self.assertEqual(detector.pending, b'')
        self.assertEqual(detector.phrase, b'')
        self.assertIsNone(detector.resampler)
        self.assertIsNone(detector.last_poll)

    def test_missing_model_is_controlled_without_network(self):
        with patch('voice_interface.vad.MODEL_PATH', Path('/missing/silero_vad.onnx')):
            with self.assertRaises(SttError):
                SpeechDetector()

    def test_real_model_accepts_silence_and_does_not_emit_recordings(self):
        detector = SpeechDetector()
        self.assertEqual(detector.window_bytes, 1024)
        for tick in range(20):
            self.assertEqual(detector.feed(bytes(9600), tick * 0.05), [])
        self.assertFalse(detector.active)
        detector.reset()

    def test_resampler_keeps_samples_across_many_reads(self):
        data = stereo(np.arange(2000, dtype='<i2'))
        detector = self.detector(itertools.repeat(False))
        model = detector.model
        for offset in range(0, len(data), 997):
            detector.feed(data[offset:offset + 997], 0)
        actual = np.concatenate([call.args[0] for call in model.is_speech.call_args_list])
        mono = audioop.tomono(data, 2, 0.5, 0.5)
        expected, _ = audioop.ratecv(mono, 2, 1, 48000, 16000, None)
        np.testing.assert_array_equal(actual, np.frombuffer(expected[:len(actual) * 2], '<i2') / 32768)
