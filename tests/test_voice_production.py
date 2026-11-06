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


AVAILABLE = all(importlib.util.find_spec(name) for name in ('discord', 'sherpa_onnx', 'pydub'))


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
