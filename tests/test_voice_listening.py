"""Playback/receive exclusion against the pinned SDK, without network calls."""
import asyncio
from contextlib import chdir, redirect_stdout
import importlib.util
import io
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

AVAILABLE = all(importlib.util.find_spec(name) for name in
                ('discord', 'sherpa_onnx', 'gtts'))
if AVAILABLE:
    import discord
    from modules.ytdl import Music
    from voice_interface.core import VoiceCommandInterface
    from voice_interface.sinks import StreamSink
    from test_voice_receive import voice_client

    class ControlledSource(discord.AudioSource):
        def __init__(self):
            self.ended = threading.Event()
            self.reading = threading.Event()
            self.error = None

        def read(self):
            self.reading.set()
            if self.error:
                raise self.error
            return b'' if self.ended.is_set() else bytes(3840)

        def cleanup(self):
            self.ended.set()


@unittest.skipUnless(AVAILABLE, 'Requires bot runtime dependencies')
class ListeningTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.enterContext(redirect_stdout(io.StringIO()))
        self.source = ControlledSource()
        self.enterContext(patch('modules.ytdl.StreamingTTS', return_value=self.source))
        self.vc = voice_client()
        self.vc.loop = asyncio.get_running_loop()
        self.vc.client.loop = self.vc.loop
        self.vc._connection.ws = SimpleNamespace(speak=AsyncMock())
        self.vc.encoder = Mock()
        self.vc.send_audio_packet = Mock()
        self.guild = object()
        self.music = Music(None, None)
        self.music.guild_voice_client[self.guild] = self.vc
        self.message = SimpleNamespace(guild=self.guild, content='!record',
                                       author=SimpleNamespace(voice=object()))

    async def asyncTearDown(self):
        player = self.vc._player
        self.vc.stop()
        if player:
            await asyncio.to_thread(player.join, 3)
            self.assertFalse(player.is_alive())

    async def started(self):
        self.assertTrue(await asyncio.to_thread(self.source.reading.wait, 3))

    def pcm(self, sink, payload):
        sender = discord.Object(id=42)
        sink.write(SimpleNamespace(pcm=payload), sender)

    async def test_default_gate_drops_pcm_and_batch_then_accepts_new_speech(self):
        sink = StreamSink()
        # Music keys guild objects; use an identity-hashable replacement.
        self.message.guild = type('Guild', (), {'id': 10})()
        self.music.guild_voice_client[self.message.guild] = self.vc
        session = SimpleNamespace(closed=False, fallback=False,
            reset_detector=AsyncMock(), detect=AsyncMock(return_value=[b'one', b'two']),
            transcribe=AsyncMock(return_value='question'))
        chat = SimpleNamespace(send_message=AsyncMock(return_value='answer'))
        interface = VoiceCommandInterface(None, session, chat, self.music)
        with tempfile.TemporaryDirectory() as directory, chdir(directory), \
                patch('discord.ext.tasks.Loop.start'):
            loop = await interface.start_listening(self.message, sink, 42)
            self.pcm(sink, b'question')
            pending = asyncio.create_task(loop.coro(self.message))
            await self.started()
            self.pcm(sink, b'must be dropped')
            self.assertEqual(sink.read(42), b'')
            self.assertFalse(pending.done())
            self.assertEqual(session.detect.await_count, 1)
            self.vc.pause()
            self.pcm(sink, b'paused too')
            self.assertEqual(sink.read(42), b'')
            self.assertFalse(pending.done())
            self.vc.resume()
            self.source.ended.set()
            await asyncio.wait_for(pending, 3)
            self.assertFalse(list(Path('.').glob('*.wav')))
            session.transcribe.assert_awaited_once()
            self.assertEqual(session.reset_detector.await_count, 3)
            session.detect.return_value = []
            self.pcm(sink, b'new question')
            await loop.coro(self.message)
            self.assertEqual(session.detect.call_args.args[0], b'new question')

    async def test_switch_true_restores_nonwaiting_playback_and_retains_pcm(self):
        sink = StreamSink()
        self.message.guild = type('Guild', (), {'id': 10})()
        session = SimpleNamespace(closed=False, detect=AsyncMock(return_value=[b'one', b'two']),
                                  reset_detector=AsyncMock())
        interface = VoiceCommandInterface(None, session, None, None)

        async def process(message, filename, *, sink):
            self.assertIsNone(sink)
            await self.music.say(self.message, 'answer')
            Path(filename).unlink()

        # Only the first WAV starts playback; verify that the second is still processed.
        self.music.guild_voice_client[self.message.guild] = self.vc
        with tempfile.TemporaryDirectory() as directory, chdir(directory), \
                patch('discord.ext.tasks.Loop.start'), \
                patch.object(interface, '_process_recording', side_effect=process) as handle:
            loop = await interface.start_listening(self.message, sink, 42,
                                                   listen_while_speaking=True)
            await asyncio.wait_for(loop.coro(self.message), 3)
            self.assertEqual(handle.await_count, 2)
            await self.started()
            self.pcm(sink, b'retained')
            self.assertEqual(sink.read(42), b'retained')
            session.reset_detector.assert_awaited_once()

    async def test_cancel_paused_tts_keeps_receive_reader_and_stops_own_player(self):
        reader = Mock()
        self.vc._reader = reader
        pending = asyncio.create_task(self.music.say(self.message, 'answer', wait=True))
        await self.started()
        player = self.vc._player
        self.vc.pause()
        pending.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(pending, 3)
        await asyncio.to_thread(player.join, 3)
        self.assertFalse(player.is_alive())
        self.assertTrue(self.source.ended.is_set())
        reader.stop.assert_not_called()

    async def test_sdk_stop_disconnect_and_playback_error_finish_wait(self):
        for ending in ('stop', 'disconnect', 'error'):
            with self.subTest(ending=ending):
                self.source = ControlledSource()
                with patch('modules.ytdl.StreamingTTS', return_value=self.source):
                    pending = asyncio.create_task(self.music.say(self.message, 'answer', wait=True))
                    await self.started()
                    player = self.vc._player
                    if ending == 'stop':
                        self.vc.stop()
                    elif ending == 'disconnect':
                        self.vc._connection.disconnect = AsyncMock()
                        with patch.object(self.vc, 'cleanup'):
                            await self.vc.disconnect()
                    else:
                        self.source.error = RuntimeError('fake player error')
                    await asyncio.wait_for(pending, 3)
                    await asyncio.to_thread(player.join, 3)
                    self.assertFalse(player.is_alive())

    async def test_core_restores_sink_after_play_failure_and_cancellation(self):
        for cancel in (False, True):
            with self.subTest(cancel=cancel):
                sink = StreamSink()
                self.pcm(sink, b'before')
                entered = asyncio.Event()

                async def fail(*args, **kwargs):
                    self.assertEqual(sink.read(42), b'')
                    self.pcm(sink, b'during')
                    entered.set()
                    if cancel:
                        await asyncio.Future()
                    raise RuntimeError('play rejected')

                session = SimpleNamespace(closed=False, fallback=False,
                    reset_detector=AsyncMock(), transcribe=AsyncMock(return_value='question'))
                interface = VoiceCommandInterface(None, session,
                    SimpleNamespace(send_message=AsyncMock(return_value='answer')),
                    SimpleNamespace(say=fail))
                with tempfile.TemporaryDirectory() as directory:
                    filename = Path(directory) / 'phrase.wav'
                    filename.touch()
                    pending = asyncio.create_task(interface._process_recording(
                        self.message, str(filename), sink=sink))
                    await asyncio.wait_for(entered.wait(), 3)
                    if cancel:
                        pending.cancel()
                    with self.assertRaises(asyncio.CancelledError if cancel else RuntimeError):
                        await pending
                    self.assertFalse(filename.exists())
                self.assertEqual(sink.read(42), b'')
                self.pcm(sink, b'after')
                self.assertEqual(sink.read(42), b'after')
                self.assertEqual(session.reset_detector.await_count, 2)


@unittest.skipUnless(AVAILABLE, 'Requires bot runtime dependencies')
class SinkGateTests(unittest.TestCase):
    def test_gate_is_atomic_with_writer_and_does_not_affect_other_sink(self):
        sink, other = StreamSink(), StreamSink()
        sender = discord.Object(id=42)
        stop = threading.Event()
        writing = threading.Event()

        def write():
            while not stop.is_set():
                sink.write(SimpleNamespace(pcm=b'old'), sender)
                writing.set()

        thread = threading.Thread(target=write)
        thread.start()
        try:
            self.assertTrue(writing.wait(3))
            sink.set_listening(False)
            self.assertEqual(sink.read(42), b'')
            other.write(SimpleNamespace(pcm=b'other guild'), sender)
            self.assertEqual(other.read(42), b'other guild')
        finally:
            stop.set()
            thread.join(3)
        self.assertFalse(thread.is_alive())
        self.assertEqual(sink.read(42), b'')
        sink.set_listening(True)
        sink.write(SimpleNamespace(pcm=b'new'), sender)
        self.assertEqual(sink.read(42), b'new')
        sink.cleanup()
        sink.set_listening(True)
        sink.write(SimpleNamespace(pcm=b'late'), sender)
        self.assertEqual(sink.audio_data, {})
