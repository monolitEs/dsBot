"""Offline contract tests; run from repository root with unittest discovery."""
import asyncio
from http.client import IncompleteRead
import io
import json
from pathlib import Path
import threading
import unittest
from unittest.mock import patch
import urllib.error
import wave

from dsbot.voice.cloudflare_stt import CloudflareTranscriber
from dsbot.voice.local_stt import LocalTranscriber
from dsbot.voice.transcription import QuotaExceeded, SerializedTranscriber, SttError, run_blocking


def wav_bytes(rate=16000, channels=1):
    output = io.BytesIO()
    with wave.open(output, 'wb') as audio:
        audio.setnchannels(channels)
        audio.setsampwidth(2)
        audio.setframerate(rate)
        audio.writeframes(b'\0\0' * channels * 160)
    return output.getvalue()


class CloudflareTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.backend = CloudflareTranscriber('a' * 32, 'fake-secret')
        await self.backend.start()

    async def test_request_and_both_response_formats(self):
        import base64
        for result in [{'text': ' Привет '}, {'transcription_info': {'text': ' Привет '}}]:
            response = io.BytesIO(json.dumps({'success': True, 'result': result}).encode())
            with patch('urllib.request.urlopen', return_value=response) as request:
                self.assertEqual(await self.backend.transcribe(wav_bytes()), 'Привет')
                sent = request.call_args.args[0]
                body = json.loads(sent.data)
                self.assertEqual(base64.b64decode(body['audio']), wav_bytes())
                self.assertEqual(body['language'], 'ru')
                self.assertEqual(body['task'], 'transcribe')
                self.assertEqual(sent.get_header('Authorization'), 'Bearer fake-secret')
                self.assertEqual(request.call_args.kwargs['timeout'], 45)

    async def test_only_exact_quota_is_typed_as_quota(self):
        for status, code in [(429, 3036), (429, 3040), (429, None), (401, 3036), (403, 5035)]:
            body = json.dumps({'errors': [{'code': code, 'message': 'fake-secret'}]}).encode()
            error = urllib.error.HTTPError('https://fake-secret', status, 'fake-secret', {}, io.BytesIO(body))
            with patch('urllib.request.urlopen', side_effect=error) as request:
                with self.assertRaises(SttError) as caught:
                    await self.backend.transcribe(wav_bytes())
                self.assertEqual(isinstance(caught.exception, QuotaExceeded), (status, code) == (429, 3036))
                self.assertNotIn('fake-secret', str(caught.exception))
                self.assertEqual(request.call_count, 1)

    async def test_malformed_error_body_is_sanitized(self):
        for body in [b'fake-secret', b'[]', b'{"errors":null}']:
            error = urllib.error.HTTPError('secret', 429, 'fake-secret', {}, io.BytesIO(body))
            with patch('urllib.request.urlopen', side_effect=error):
                with self.assertRaises(SttError) as caught:
                    await self.backend.transcribe(wav_bytes())
                self.assertNotIsInstance(caught.exception, QuotaExceeded)
                self.assertNotIn('fake-secret', str(caught.exception))

    async def test_schema_network_and_empty_text(self):
        for payload in [[], {'success': False}, {'success': True, 'result': None},
                        {'success': True, 'result': {'text': 123}}]:
            with patch('urllib.request.urlopen', return_value=io.BytesIO(json.dumps(payload).encode())):
                with self.assertRaises(SttError):
                    await self.backend.transcribe(wav_bytes())
        for error in [urllib.error.URLError('fake-secret'), TimeoutError('fake-secret'),
                      IncompleteRead(b'fake-secret')]:
            with patch('urllib.request.urlopen', side_effect=error):
                with self.assertRaises(SttError) as caught:
                    await self.backend.transcribe(wav_bytes())
                self.assertNotIn('fake-secret', str(caught.exception))
        for body in [b'fake-secret', b'\xff']:
            with patch('urllib.request.urlopen', return_value=io.BytesIO(body)):
                with self.assertRaises(SttError) as caught:
                    await self.backend.transcribe(wav_bytes())
                self.assertNotIn('fake-secret', str(caught.exception))
        with patch('urllib.request.urlopen', return_value=io.BytesIO(b'{"success":true,"result":{"text":" "}}')):
            self.assertEqual(await self.backend.transcribe(wav_bytes()), '')

    async def test_invalid_audio_never_sent(self):
        with patch('urllib.request.urlopen') as request:
            for audio in [b'not a wav', wav_bytes(rate=48000), wav_bytes(channels=2), wav_bytes()[:-5]]:
                with self.assertRaises(SttError):
                    await self.backend.transcribe(audio)
            request.assert_not_called()

    async def test_lifecycle_and_bad_config(self):
        await self.backend.close()
        with self.assertRaises(SttError):
            await self.backend.transcribe(wav_bytes())
        await self.backend.start()
        await self.backend.close()
        for account, token, timeout in [('bad', 'key', 45), ('a'*32, 'key\n', 45),
                                        ('a'*32, '', 45), ('a'*32, 'key', 0), ('a'*32, 'key', float('nan'))]:
            with self.assertRaises(ValueError):
                CloudflareTranscriber(account, token, timeout=timeout)


class LifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_package_export_remains_available_without_eager_core_import(self):
        import sys
        import types
        import dsbot.voice
        fake_core = types.ModuleType('dsbot.voice.core')
        fake_core.VoiceCommandInterface = object()
        with patch.dict(sys.modules, {'dsbot.voice.core': fake_core}):
            self.assertIs(dsbot.voice.VoiceCommandInterface, fake_core.VoiceCommandInterface)
        with self.assertRaises(AttributeError):
            getattr(dsbot.voice, 'unknown_name')

    async def test_internal_task_cancellation_does_not_spin_forever(self):
        with patch('asyncio.to_thread', side_effect=asyncio.CancelledError):
            with self.assertRaises(asyncio.CancelledError):
                await run_blocking(lambda: None)

    async def test_trim_is_optional_and_native_reference_is_released(self):
        backend = LocalTranscriber('unused')
        backend._model = object()
        with patch('dsbot.voice.local_stt.sys.platform', 'linux'):
            with patch('dsbot.voice.local_stt.ctypes.CDLL', side_effect=OSError):
                await asyncio.to_thread(backend._unload)
        self.assertIsNone(backend._model)

    async def test_concurrent_calls_are_serial_and_close_is_idempotent(self):
        class Backend(SerializedTranscriber):
            active = 0
            max_active = 0
            loads = 0
            unloads = 0

            def _load(self):
                self.loads += 1

            def _unload(self):
                self.unloads += 1

            def _transcribe(self, wav):
                import time
                self.active += 1
                self.max_active = max(self.max_active, self.active)
                time.sleep(0.01)
                self.active -= 1
                return 'text'

        backend = Backend()
        await asyncio.gather(backend.start(), backend.start())
        self.assertEqual(await asyncio.gather(*(backend.transcribe(wav_bytes()) for _ in range(3))), ['text']*3)
        await asyncio.gather(backend.close(), backend.close())
        self.assertEqual((backend.max_active, backend.loads, backend.unloads), (1, 1, 1))

    async def test_cancelled_inference_keeps_lock_until_thread_finishes(self):
        started, finish = threading.Event(), threading.Event()

        class Backend(SerializedTranscriber):
            unloaded = False

            def _transcribe(self, wav):
                started.set()
                if not finish.wait(5):
                    raise RuntimeError('test thread timed out')
                self.assert_unloaded = self.unloaded
                return 'late result'

            def _unload(self):
                self.unloaded = True

        backend = Backend()
        await backend.start()
        pending = asyncio.create_task(backend.transcribe(wav_bytes()))
        try:
            self.assertTrue(await asyncio.to_thread(started.wait, 5))
            pending.cancel()
            await asyncio.sleep(0)
            pending.cancel()  # Repeated cancellation must not release the native-model lock.
            closing = asyncio.create_task(backend.close())
            await asyncio.sleep(0)
            self.assertFalse(closing.done())
            self.assertFalse(backend.unloaded)
        finally:
            finish.set()
        with self.assertRaises(asyncio.CancelledError):
            await pending
        await closing
        self.assertFalse(backend.assert_unloaded)
        self.assertTrue(backend.unloaded)

    async def test_cancelled_loading_is_unloaded_and_can_restart(self):
        entered, finish = threading.Event(), threading.Event()

        class Backend(SerializedTranscriber):
            loaded = False

            def _load(self):
                entered.set()
                if not finish.wait(5):
                    raise RuntimeError('test thread timed out')
                self.loaded = True

            def _unload(self):
                self.loaded = False

            def _transcribe(self, wav):
                return 'text'

        backend = Backend()
        loading = asyncio.create_task(backend.start())
        try:
            self.assertTrue(await asyncio.to_thread(entered.wait, 5))
            loading.cancel()
        finally:
            finish.set()
        with self.assertRaises(asyncio.CancelledError):
            await loading
        self.assertFalse(backend.loaded)
        await backend.start()
        self.assertTrue(backend.loaded)
        await backend.close()

    async def test_local_import_is_lazy_and_missing_model_is_controlled(self):
        backend = LocalTranscriber(Path('missing-test-model'))
        with patch.dict('sys.modules', {'sherpa_onnx': None}):
            with self.assertRaises(SttError):
                await backend.start()
        await backend.close()


if __name__ == '__main__':
    unittest.main()
