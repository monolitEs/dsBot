"""Pinned SDK integrity and finite-utterance regression tests; no network."""
import hashlib
import importlib
import importlib.metadata
import importlib.util
import io
import json
from pathlib import Path
import unittest
from unittest.mock import Mock, patch


AVAILABLE = all(importlib.util.find_spec(name) for name in ('discord', 'vosk', 'pydub'))


class WheelIntegrityTests(unittest.TestCase):
    def test_committed_wheel_matches_manifest_and_install_hash(self):
        root = Path(__file__).resolve().parents[1]
        vendor = root / 'docker/vendor/pycord'
        record = json.loads((vendor / 'wheel-provenance.json').read_text())
        digest = hashlib.sha256((vendor / record['file']).read_bytes()).hexdigest()
        self.assertEqual(digest, record['sha256'])
        self.assertIn(f"{record['file']}#sha256={digest}",
                      (root / 'requirements.txt').read_text())


@unittest.skipUnless(AVAILABLE, 'Requires bot runtime dependencies')
class ProductionVoiceTests(unittest.TestCase):
    def test_exact_installed_versions_and_voice_subpackages(self):
        self.assertEqual(importlib.metadata.version('py-cord'), '2.8.1.dev0+g10a5e8c')
        self.assertEqual(importlib.metadata.version('davey'), '0.1.6')
        for name in ('receive', 'packets', 'utils'):
            importlib.import_module(f'discord.voice.{name}')

    def detect(self, accepted, result, final):
        from pydub import AudioSegment
        from voice_interface import stt
        recognizer = Mock()
        recognizer.AcceptWaveform.return_value = accepted
        recognizer.Result.return_value = json.dumps({'text': result})
        recognizer.FinalResult.return_value = json.dumps({'text': final})
        wav = AudioSegment.silent(duration=100, frame_rate=16000).export(io.BytesIO(), format='wav')
        with patch.object(stt, 'KaldiRecognizer', return_value=recognizer):
            text = stt.detect_words(wav, object())
        return text, recognizer

    def test_incomplete_phrase_uses_final_without_duplicate_partial(self):
        text, recognizer = self.detect(False, 'duplicate partial', 'finished words')
        self.assertEqual(text, 'finished words')
        recognizer.Result.assert_not_called()
        recognizer.FinalResult.assert_called_once_with()

    def test_complete_phrase_keeps_result_and_final_tail(self):
        text, recognizer = self.detect(True, 'first words', 'last words')
        self.assertEqual(text, 'first words last words')
        recognizer.Result.assert_called_once_with()
        recognizer.FinalResult.assert_called_once_with()

    def test_empty_tail_preserves_completed_words_and_silence_stays_empty(self):
        self.assertEqual(self.detect(True, 'finished words', '')[0], 'finished words')
        self.assertEqual(self.detect(False, '', '')[0], '')
