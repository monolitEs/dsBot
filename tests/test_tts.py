"""Streaming latency, bounded buffering and lifecycle tests; no TTS network."""
import asyncio
from contextlib import redirect_stdout
import importlib.util
import io
import shutil
import subprocess
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

AVAILABLE = all(importlib.util.find_spec(name) for name in ('discord', 'gtts', 'yt_dlp'))
if AVAILABLE:
    import discord
    from discord.player import AudioPlayer
    from dsbot.music.tts import StreamingTTS
    from dsbot.music.player import Music


@unittest.skipUnless(AVAILABLE, 'Requires bot runtime dependencies')
class StreamTests(unittest.TestCase):
    def setUp(self):
        self.output = self.enterContext(redirect_stdout(io.StringIO()))

    def wait_for(self, condition):
        deadline = time.monotonic() + 3
        while not condition():
            if time.monotonic() > deadline:
                self.fail('Worker did not reach the expected state')
            time.sleep(0.001)

    def finish(self, source):
        source.cleanup()
        if source._started:
            source._worker.join(3)
            self.assertFalse(source._worker.is_alive(), 'Leaked producer thread')

    def start(self, source):
        source._started = True
        source._worker.start()

    def chunks(self, parts):
        yield from parts

    def source(self, chunks, decoded=None):
        factory = self.enterContext(patch('dsbot.music.tts.gTTS'))
        factory.return_value.stream.return_value = self.chunks(chunks)
        if decoded is not None:
            self.enterContext(patch.object(StreamingTTS, '_decode', side_effect=decoded))
        source = StreamingTTS('synthetic test')
        self.addCleanup(self.finish, source)
        return source, factory

    def test_first_audio_precedes_last_chunk_and_starvation_is_not_eof(self):
        release = threading.Event()
        waiting = threading.Event()

        def chunks():
            yield b'first'
            waiting.set()
            release.wait(3)
            yield b'last'

        frame = b'\1' * StreamingTTS.FRAME_SIZE
        source, factory = self.source(chunks(), [frame, frame])
        self.addCleanup(release.set)
        self.start(source)
        self.assertTrue(waiting.wait(3))
        self.assertFalse(source._done.is_set())
        self.assertEqual(source.read(), frame)
        self.assertEqual(source.read(), source.SILENCE)
        release.set()
        self.assertTrue(source._done.wait(3))
        self.assertEqual(source.read(), frame)
        self.assertEqual(source.read(), b'')
        factory.assert_called_once_with('synthetic test', lang='ru', timeout=(5, 15))
        self.assertIn('Первый фрагмент передан плееру:', self.output.getvalue())
        self.assertIn('Загрузка и подготовка всей речи:', self.output.getvalue())

    def test_partial_frames_are_joined_in_order_and_only_final_tail_is_padded(self):
        size = StreamingTTS.FRAME_SIZE
        source, _ = self.source([b'a', b'b'], [b'\1' * (size + 4), b'\2' * 4])
        self.start(source)
        self.assertTrue(source._done.wait(3))
        self.assertEqual(source.read(), b'\1' * size)
        self.assertEqual(source.read(), (b'\1' * 4 + b'\2' * 4).ljust(size, b'\0'))
        self.assertEqual(source.read(), b'')

    def test_full_buffer_limits_prefetch_and_cleanup_releases_backpressure(self):
        advanced = threading.Event()

        def chunks():
            yield b'first'
            advanced.set()
            yield b'last'

        source, _ = self.source(chunks(), [b'\1' * StreamingTTS.FRAME_SIZE * 100])
        self.start(source)
        self.wait_for(source._frames.full)
        self.assertLessEqual(source._frames.qsize(), 50)
        self.assertFalse(advanced.is_set())
        self.finish(source)
        self.assertFalse(advanced.is_set())
        self.assertEqual(source.read(), b'')

    def test_stop_during_http_discards_response_and_does_not_request_next(self):
        release = threading.Event()
        requested = threading.Event()
        advanced = threading.Event()

        def chunks():
            requested.set()
            release.wait(3)
            yield b'inflight'
            advanced.set()
            yield b'unwanted'

        source, _ = self.source(chunks(), [])
        self.addCleanup(release.set)
        self.start(source)
        self.assertTrue(requested.wait(3))
        source.cleanup()
        self.assertEqual(source.read(), b'')
        release.set()
        self.assertTrue(source._done.wait(3))
        self.assertFalse(advanced.is_set())
        self.assertEqual(self.output.getvalue(), '')

    def test_network_error_before_or_after_audio_ends_without_replay(self):
        for partial in (False, True):
            with self.subTest(partial=partial):
                def chunks():
                    if partial:
                        yield b'first'
                    raise OSError('sensitive upstream response')

                source, _ = self.source(chunks(), [b'\1' * StreamingTTS.FRAME_SIZE])
                self.start(source)
                self.assertTrue(source._done.wait(3))
                if partial:
                    self.assertEqual(source.read(), b'\1' * StreamingTTS.FRAME_SIZE)
                self.assertEqual(source.read(), b'')
                self.assertNotIn('sensitive', self.output.getvalue())
                self.assertIn('Ошибка озвучки: OSError', self.output.getvalue())

    def test_decode_error_and_empty_stream_finish(self):
        for chunks, decoded in (([b'invalid'], [RuntimeError('bad audio')]), ([], [])):
            source, _ = self.source(chunks, decoded)
            self.start(source)
            self.assertTrue(source._done.wait(3))
            self.assertEqual(source.read(), b'')

    def test_two_sources_have_independent_buffers_and_cancellation(self):
        factory = self.enterContext(patch('dsbot.music.tts.gTTS'))
        factory.return_value.stream.side_effect = [self.chunks([b'one']), self.chunks([b'two'])]
        self.enterContext(patch.object(StreamingTTS, '_decode',
                                       side_effect=lambda data: data[:1] * StreamingTTS.FRAME_SIZE))
        one, two = StreamingTTS('one'), StreamingTTS('two')
        self.addCleanup(self.finish, one)
        self.addCleanup(self.finish, two)
        self.start(one)
        self.assertTrue(one._done.wait(3))
        self.start(two)
        self.assertTrue(two._done.wait(3))
        one.cleanup()
        self.assertEqual(one.read(), b'')
        self.assertEqual(two.read(), b't' * StreamingTTS.FRAME_SIZE)

    def test_cleanup_before_playback_never_starts_http(self):
        source, factory = self.source([], [])
        source.cleanup()
        source.cleanup()
        self.assertEqual(source.read(), b'')
        self.assertFalse(source._started)
        factory.assert_not_called()

    def test_cleanup_kills_active_decoder_and_worker_reaps_it(self):
        entered = threading.Event()
        released = threading.Event()
        decoder = Mock()
        decoder.__enter__ = Mock(return_value=decoder)
        decoder.__exit__ = Mock(return_value=False)
        decoder.poll.return_value = None
        decoder.returncode = -9

        def communicate(_):
            entered.set()
            released.wait(3)
            return b'', None

        decoder.communicate.side_effect = communicate
        decoder.kill.side_effect = released.set
        self.enterContext(patch('dsbot.music.tts.subprocess.Popen', return_value=decoder))
        source, _ = self.source([b'audio'])
        self.addCleanup(released.set)
        self.start(source)
        self.assertTrue(entered.wait(3))
        self.finish(source)
        decoder.kill.assert_called_once()
        decoder.__exit__.assert_called_once()
        self.assertEqual(self.output.getvalue(), '')

    @unittest.skipUnless(shutil.which('ffmpeg'), 'Requires FFmpeg')
    def test_real_ffmpeg_decodes_mp3_parts_before_last_is_available(self):
        mp3 = subprocess.check_output([
            'ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'sine=frequency=440:duration=0.1',
            '-f', 'mp3', 'pipe:1'])
        release = threading.Event()
        waiting = threading.Event()

        def chunks():
            yield mp3
            waiting.set()
            release.wait(3)
            yield mp3

        source, _ = self.source(chunks())
        self.addCleanup(release.set)
        self.start(source)
        self.assertTrue(waiting.wait(3))
        first = source.read()
        self.assertEqual(len(first), source.FRAME_SIZE)
        self.assertNotEqual(first, source.SILENCE)
        self.assertFalse(source._done.is_set())
        release.set()
        self.assertTrue(source._done.wait(3))
        frames = []
        while data := source.read():
            frames.append(data)
        self.assertTrue(frames)
        self.assertTrue(all(len(frame) == source.FRAME_SIZE for frame in frames))


@unittest.skipUnless(AVAILABLE, 'Requires bot runtime dependencies')
class PlaybackTests(unittest.IsolatedAsyncioTestCase):
    async def test_busy_playing_or_paused_voice_does_not_create_tts(self):
        for playing, paused in ((True, False), (False, True)):
            guild = object()
            message = SimpleNamespace(guild=guild, author=SimpleNamespace(voice=object()))
            music = Music(None, None)
            music.guild_voice_client[guild] = Mock(
                is_playing=Mock(return_value=playing), is_paused=Mock(return_value=paused))
            with patch('dsbot.music.player.StreamingTTS') as factory, redirect_stdout(io.StringIO()):
                await music.say(message, 'answer')
            factory.assert_not_called()

    async def test_play_rejection_cleans_source_without_network_request(self):
        guild = object()
        message = SimpleNamespace(guild=guild, author=SimpleNamespace(voice=object()))
        music = Music(None, None)
        music.guild_voice_client[guild] = Mock(
            is_playing=Mock(return_value=False), is_paused=Mock(return_value=False),
            play=Mock(side_effect=RuntimeError('disconnected')))
        with patch('dsbot.music.player.StreamingTTS', wraps=StreamingTTS), \
                patch('dsbot.music.tts.gTTS') as upstream:
            with self.assertRaises(RuntimeError):
                await music.say(message, 'answer')
            upstream.assert_not_called()

    async def test_pinned_player_stop_and_disconnect_cleanup_while_http_waits(self):
        for paused, disconnect in ((False, False), (True, False), (False, True)):
            release = threading.Event()
            requested = threading.Event()

            def chunks():
                requested.set()
                release.wait(3)
                yield b'not decoded after stop'

            with patch('dsbot.music.tts.gTTS') as upstream:
                upstream.return_value.stream.return_value = chunks()
                source = StreamingTTS('test')
                client = Mock()
                client.loop = asyncio.get_running_loop()
                client.client.loop = client.loop
                client.ws.speak = AsyncMock()
                player = AudioPlayer(source, client)
                player.start()
                try:
                    self.assertTrue(await asyncio.to_thread(requested.wait, 3))
                    if paused:
                        player.pause()
                    if disconnect:
                        vc = object.__new__(discord.voice.VoiceClient)
                        vc._player = player
                        vc._player_future = None
                        from discord.utils import MISSING
                        vc._reader = MISSING
                        vc._connection = SimpleNamespace(disconnect=AsyncMock())
                        vc.cleanup = Mock()
                        await vc.disconnect()
                        vc._connection.disconnect.assert_awaited_once()
                    else:
                        player.stop()
                    await asyncio.to_thread(player.join, 3)
                    self.assertFalse(player.is_alive())
                    self.assertTrue(source._stopped.is_set())
                finally:
                    player.stop()
                    release.set()
                    await asyncio.to_thread(player.join, 3)
                    await asyncio.to_thread(source._worker.join, 3)
                    self.assertFalse(source._worker.is_alive())
