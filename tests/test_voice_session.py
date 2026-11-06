"""Voice orchestration tests; Pycord/pydub/Silero tests run in the bot image."""
import asyncio
from contextlib import chdir, redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

AVAILABLE = all(importlib.util.find_spec(name) for name in ['discord', 'pydub', 'sherpa_onnx'])
if AVAILABLE:
    import bot as app
    from modules.ytdl import Music
    from voice_interface.core import VoiceCommandInterface
    from voice_interface.stt import VoiceSession
    from voice_interface.transcription import QuotaExceeded, SttError


@unittest.skipUnless(AVAILABLE, 'Requires bot runtime dependencies')
class SessionTests(unittest.IsolatedAsyncioTestCase):
    async def test_idle_listener_and_repeated_close_produce_no_logs(self):
        session = self.make_session()
        session.detect = AsyncMock(return_value=[])
        message = SimpleNamespace(guild=SimpleNamespace(id=10))
        sink = SimpleNamespace(read=Mock(side_effect=[KeyError(), b'', b'\0' * 1920]))
        interface = VoiceCommandInterface(None, session, None, None)
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as directory, chdir(directory), \
                patch('discord.ext.tasks.Loop.start'), redirect_stdout(output):
            loop = await interface.start_listening(message, sink, 7)
            for _ in range(3):
                await loop.coro(message)
            await session.close()
            await session.close()
        self.assertEqual(output.getvalue(), '')
        self.assertEqual(session.detect.await_count, 3)

    def make_session(self):
        cloud = SimpleNamespace(start=AsyncMock(), transcribe=AsyncMock(return_value='cloud'), close=AsyncMock())
        local = SimpleNamespace(start=AsyncMock(), transcribe=AsyncMock(return_value='local'), close=AsyncMock())
        return VoiceSession(cloud, local)

    async def test_load_on_start_only_and_cleanup_on_exit(self):
        session = self.make_session()
        with patch('voice_interface.stt.SpeechDetector', return_value=object()) as model:
            session.local.start.assert_not_called()
            await asyncio.gather(session.start(), session.start())
            model.assert_called_once()
            session.local.start.assert_awaited_once()
        await session.close()
        self.assertTrue(session.closed)
        self.assertIsNone(session.detector)
        session.local.close.assert_awaited_once()
        with self.assertRaises(SttError):
            await session.start()

    async def test_exact_quota_replays_same_wav_locally_for_rest_of_session(self):
        session = self.make_session()
        session.cloud.transcribe.side_effect = QuotaExceeded('quota')
        with patch('voice_interface.stt.AudioSegment.from_wav') as read:
            read.return_value.set_frame_rate.return_value.export.return_value.getvalue.return_value = b'normalized-wav'
            self.assertEqual(await session.transcribe('phrase.wav'), 'local')
            self.assertEqual(await session.transcribe('next.wav'), 'local')
        session.cloud.transcribe.assert_awaited_once_with(b'normalized-wav')
        self.assertEqual(session.local.transcribe.await_count, 2)
        session.local.transcribe.assert_awaited_with(b'normalized-wav')
        self.assertFalse(self.make_session().fallback)

    async def test_other_errors_do_not_switch_backend(self):
        session = self.make_session()
        session.cloud.transcribe.side_effect = SttError('Cloudflare HTTP 403')
        with patch('voice_interface.stt.AudioSegment.from_wav'):
            with self.assertRaises(SttError):
                await session.transcribe('phrase.wav')
        session.local.transcribe.assert_not_called()
        self.assertFalse(session.fallback)

    async def test_failed_start_rolls_back_models(self):
        session = self.make_session()
        with patch('voice_interface.stt.SpeechDetector', side_effect=RuntimeError('missing model')):
            with self.assertRaises(RuntimeError):
                await session.start()
        session.local.close.assert_awaited_once()
        session.cloud.close.assert_awaited_once()

    async def test_record_stop_keeps_models_and_exit_releases_them(self):
        session = self.make_session()
        session.detector = object()
        pending = asyncio.create_task(asyncio.sleep(10))
        session.listener = SimpleNamespace(cancel=pending.cancel, get_task=lambda: pending)
        await session.stop()
        self.assertTrue(pending.cancelled())
        self.assertIsNotNone(session.detector)
        session.local.close.assert_not_called()
        await session.close()
        self.assertIsNone(session.detector)

    async def test_no_late_chat_or_tts_after_disconnect(self):
        session = self.make_session()
        session.detect = AsyncMock(return_value=[b'fake-wav'])
        entered, finish = asyncio.Event(), asyncio.Event()
        async def transcribe(filename):
            entered.set()
            await finish.wait()
            return 'late words'
        session.transcribe = transcribe
        chat = SimpleNamespace(send_message=AsyncMock(return_value='answer'))
        music = SimpleNamespace(say=AsyncMock())
        message = SimpleNamespace(guild=SimpleNamespace(id=10), content='!record')
        interface = VoiceCommandInterface(None, session, chat, music)
        sink = SimpleNamespace(read=Mock(return_value=b''))
        with tempfile.TemporaryDirectory() as directory, chdir(directory), \
                patch('discord.ext.tasks.Loop.start'):
            loop = await interface.start_listening(message, sink, 7)
            pending = asyncio.create_task(loop.coro(message))
            await asyncio.wait_for(entered.wait(), 2)
            session.closed = True
            finish.set()
            await pending
            self.assertFalse(list(Path('.').glob('*.wav')))
        chat.send_message.assert_not_called()
        music.say.assert_not_called()
        self.assertEqual(message.content, '!record')


    async def test_exit_waits_for_actual_http_thread_before_freeing_models(self):
        from voice_interface.cloudflare_stt import CloudflareTranscriber
        from test_transcription import wav_bytes
        session = self.make_session()
        session.cloud = CloudflareTranscriber('a' * 32, 'fake-key')
        await session.cloud.start()
        entered, finish = threading.Event(), threading.Event()

        def respond(*args, **kwargs):
            entered.set()
            if not finish.wait(5):
                raise RuntimeError('test timeout')
            return io.BytesIO(b'{"success":true,"result":{"text":"late"}}')

        with patch('voice_interface.stt.AudioSegment.from_wav') as read, \
                patch('urllib.request.urlopen', side_effect=respond):
            read.return_value.set_frame_rate.return_value.export.return_value.getvalue.return_value = wav_bytes()
            pending = asyncio.create_task(session.transcribe('phrase.wav'))
            session.listener = SimpleNamespace(cancel=pending.cancel, get_task=lambda: pending)
            try:
                self.assertTrue(await asyncio.to_thread(entered.wait, 5))
                closing = asyncio.create_task(session.close())
                await asyncio.sleep(0.02)
                self.assertTrue(session.closed)
                self.assertFalse(closing.done())
                session.local.close.assert_not_called()
            finally:
                finish.set()
            await closing
        self.assertTrue(pending.cancelled())
        session.local.close.assert_awaited_once()

    async def test_close_drains_vad_thread_before_reset_and_unload(self):
        session = self.make_session()
        entered, finish = threading.Event(), threading.Event()

        def infer(*args):
            entered.set()
            if not finish.wait(5):
                raise RuntimeError('VAD test timed out')
            return []

        detector = Mock()
        detector.feed.side_effect = infer
        session.detector = detector
        pending = asyncio.create_task(session.detect(b'pcm', 0))
        session.listener = SimpleNamespace(cancel=pending.cancel, get_task=lambda: pending)
        try:
            self.assertTrue(await asyncio.to_thread(entered.wait, 2))
            closing = asyncio.create_task(session.close())
            await asyncio.sleep(0.02)
            self.assertFalse(closing.done())
            self.assertIs(session.detector, detector)
            session.local.close.assert_not_called()
        finally:
            finish.set()
        await closing
        self.assertTrue(pending.cancelled())
        self.assertIsNone(session.detector)

    async def test_pcm_to_cloud_text_then_chat_and_speech_without_changing_message(self):
        from voice_interface.cloudflare_stt import CloudflareTranscriber
        from voice_interface.transcription import read_pcm
        session = self.make_session()
        session.cloud = CloudflareTranscriber('a' * 32, 'fake-key')
        await session.cloud.start()
        from test_transcription import wav_bytes
        session.detect = AsyncMock(side_effect=[[], [wav_bytes()]])
        message = SimpleNamespace(guild=SimpleNamespace(id=10), content='!record')
        chat = SimpleNamespace(send_message=AsyncMock(return_value='answer'))
        music = SimpleNamespace(say=AsyncMock())
        interface = VoiceCommandInterface(None, session, chat, music)
        sink = SimpleNamespace(read=Mock(side_effect=[b'\0' * 1920, b'']))
        response = io.BytesIO(b'{"success":true,"result":{"text":"hello"}}')
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as directory, chdir(directory), \
                patch('discord.ext.tasks.Loop.start'), patch('urllib.request.urlopen', return_value=response) as http, \
                redirect_stdout(output):
            loop = await interface.start_listening(message, sink, 7)
            await loop.coro(message)
            http.assert_not_called()  # Keep collecting until VAD returns a complete phrase.
            await loop.coro(message)
            import base64
            payload = json.loads(http.call_args.args[0].data)
            self.assertTrue(read_pcm(base64.b64decode(payload['audio'])))
            self.assertEqual(list(Path('.').glob('*.wav')), [])
        request = chat.send_message.call_args.args[0]
        self.assertEqual(request.content, '!hello')
        self.assertEqual(message.content, '!record')
        music.say.assert_awaited_once_with(request, 'answer')
        lines = [line for line in output.getvalue().splitlines() if not line.startswith('VAD:')]
        self.assertEqual(len(lines), 4)
        self.assertRegex(lines[0], r'^STT: Время преобразования голоса в текст: \d+\.\d{3} с$')
        self.assertEqual(lines[1], 'STT (Cloudflare): hello')
        self.assertRegex(lines[2], r'^LLM: Время ответа: \d+\.\d{3} с$')
        self.assertEqual(lines[3], 'LLM: answer')
        self.assertNotIn('detected words', output.getvalue())

    async def test_empty_stt_stops_before_chat_and_fallback_uses_correct_label(self):
        session = self.make_session()
        session.transcribe = AsyncMock(side_effect=['', 'local words'])
        session.fallback = True
        message = SimpleNamespace(content='!record')
        chat = SimpleNamespace(send_message=AsyncMock(return_value='local answer'))
        music = SimpleNamespace(say=AsyncMock())
        interface = VoiceCommandInterface(None, session, chat, music)
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(output):
            merged = Path(directory) / 'merged.wav'
            merged.touch()
            await interface._process_recording(message, str(merged))
            chat.send_message.assert_not_called()
            music.say.assert_not_called()
            self.assertFalse(merged.exists())
            merged.touch()
            await interface._process_recording(message, str(merged))
            self.assertFalse(merged.exists())
        self.assertIn('STT (GigaAM): local words', output.getvalue())
        self.assertNotIn('Cloudflare', output.getvalue())
        chat.send_message.assert_awaited_once()
        music.say.assert_awaited_once()

    async def test_tts_uses_stream_source_without_a_shared_audio_file(self):
        guild = object()
        message = SimpleNamespace(guild=guild, author=SimpleNamespace(voice=object()))
        music = Music(None, None)
        voice = Mock()
        voice.is_playing.return_value = False
        voice.is_paused.return_value = False
        music.guild_voice_client[guild] = voice
        output = io.StringIO()
        with patch('modules.ytdl.StreamingTTS') as source, redirect_stdout(output):
            await music.say(message, 'spoken answer')
        source.assert_called_once_with('spoken answer')
        voice.play.assert_called_once_with(source.return_value)
        self.assertEqual(output.getvalue(), '')

    async def test_closed_session_rejects_audio(self):
        session = self.make_session()
        await session.close()
        self.assertEqual(await session.detect(b'', 0), [])
        with self.assertRaises(SttError):
            await session.transcribe('no-file-needed')

    async def test_stt_failure_removes_recording_without_reply(self):
        session = self.make_session()
        session.detect = AsyncMock(return_value=[b'fake-wav'])
        session.transcribe = AsyncMock(side_effect=SttError('Cloudflare HTTP 403'))
        message = SimpleNamespace(guild=SimpleNamespace(id=10), content='!record')
        chat = SimpleNamespace(send_message=AsyncMock())
        music = SimpleNamespace(say=AsyncMock())
        interface = VoiceCommandInterface(None, session, chat, music)
        sink = SimpleNamespace(read=Mock(return_value=b''))
        with tempfile.TemporaryDirectory() as directory, chdir(directory), \
                patch('discord.ext.tasks.Loop.start'):
            loop = await interface.start_listening(message, sink, 7)
            await loop.coro(message)
            self.assertFalse(list(Path('.').glob('*.wav')))
        chat.send_message.assert_not_called()
        music.say.assert_not_called()


    async def test_stop_drains_wav_writer_before_cleaning_files(self):
        session = self.make_session()
        session.detect = AsyncMock(return_value=[b'fake-wav'])
        entered, finish = threading.Event(), threading.Event()
        loop_thread = threading.get_ident()
        writer_threads = []

        def write_wav(filename, data):
            writer_threads.append(threading.get_ident())
            entered.set()
            if not finish.wait(5):
                raise RuntimeError('writer test timed out')
            Path(filename).write_bytes(data)

        message = SimpleNamespace(guild=SimpleNamespace(id=10), content='!record')
        interface = VoiceCommandInterface(None, session, None, None)
        sink = SimpleNamespace(read=Mock(return_value=b'\0' * 1920))
        with tempfile.TemporaryDirectory() as directory, chdir(directory), \
                patch('voice_interface.core.write_wav', side_effect=write_wav, create=True):
            session.listener = await interface.start_listening(message, sink, 7, interval=60)
            try:
                self.assertTrue(await asyncio.to_thread(entered.wait, 5))
                stopping = asyncio.create_task(session.stop())
                await asyncio.sleep(0.02)
                self.assertFalse(stopping.done())
                self.assertNotEqual(writer_threads, [loop_thread])
                finish.set()
                await stopping
            finally:
                finish.set()
                await session.stop()
            self.assertFalse(list(Path('.').glob('*.wav')))

    async def test_stop_running_listener_resets_partial_phrase_but_keeps_models(self):
        session = self.make_session()
        session.detector = Mock()
        session.detector.feed.return_value = []
        entered = threading.Event()
        session.detector.feed.side_effect = lambda *args: (entered.set(), [])[1]
        message = SimpleNamespace(guild=SimpleNamespace(id=10), content='!record')
        interface = VoiceCommandInterface(None, session, None, None)
        sink = SimpleNamespace(read=Mock(return_value=b'pending'))
        with tempfile.TemporaryDirectory() as directory, chdir(directory):
            session.listener = await interface.start_listening(message, sink, 7, interval=60)
            try:
                self.assertTrue(await asyncio.to_thread(entered.wait, 2))
            finally:
                await session.stop()
            self.assertFalse(list(Path('.').glob('*.wav')))
        self.assertIsNotNone(session.detector)
        self.assertEqual(session.detector.reset.call_count, 2)
        session.local.close.assert_not_called()



@unittest.skipUnless(AVAILABLE, 'Requires bot runtime dependencies')
class BotVoiceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        app.voice_sessions.clear()

    async def test_music_join_returns_existing_and_moved_client(self):
        guild = object()
        channel = object()
        vc = SimpleNamespace(channel=channel, move_to=AsyncMock())
        message = SimpleNamespace(author=SimpleNamespace(voice=SimpleNamespace(channel=channel)), guild=guild)
        music = Music(SimpleNamespace(voice_clients=[vc]), None)
        with patch('discord.utils.get', return_value=vc):
            self.assertIs(await music.join(message), vc)
            message.author.voice.channel = object()
            self.assertIs(await music.join(message), vc)
        vc.move_to.assert_awaited_once()
        self.assertIs(music.guild_voice_client[guild], vc)

    async def test_music_join_no_author_voice_returns_none(self):
        message = SimpleNamespace(author=SimpleNamespace(voice=None), reply=AsyncMock())
        self.assertIsNone(await Music(SimpleNamespace(), None).join(message))

    async def test_voice_event_ignores_other_users_and_cleans_on_kick(self):
        session = SimpleNamespace(close=AsyncMock())
        app.voice_sessions[10] = session
        with patch.object(app.bot._connection, 'user', SimpleNamespace(id=99)), \
                patch.object(app, 'voice_assistant', True):
            await app.on_voice_state_update(SimpleNamespace(id=1), SimpleNamespace(channel=1), SimpleNamespace(channel=None))
            session.close.assert_not_called()
            await app.on_voice_state_update(SimpleNamespace(id=99, guild=SimpleNamespace(id=10)),
                                            SimpleNamespace(channel=1), SimpleNamespace(channel=None))
        session.close.assert_awaited_once()
        self.assertNotIn(10, app.voice_sessions)

    async def test_repeated_record_does_not_start_second_listener(self):
        vc = SimpleNamespace(is_recording=lambda: True, is_connected=lambda: True)
        session = SimpleNamespace(record_lock=asyncio.Lock(), closed=False)
        message = SimpleNamespace(reply=AsyncMock(), guild=SimpleNamespace(voice_client=vc))
        with patch.object(app, 'music', SimpleNamespace(join=AsyncMock(return_value=vc))), \
                patch.object(app, 'get_voice_session', AsyncMock(return_value=session)):
            await app.start_recording(message)
        message.reply.assert_awaited_once()

    async def test_init_true_does_not_load_models_until_voice_connection(self):
        with patch.object(app, 'Chat'), patch.object(app, 'Music'), patch.object(app, 'RoleManager'), \
                patch('voice_interface.stt.SpeechDetector') as model:
            app.init(True)
            self.assertTrue(app.voice_assistant)
            self.assertFalse(app.voice_sessions)
            model.assert_not_called()
            app.init(False)

    async def test_session_is_reused_and_replaced_after_disconnect(self):
        first = SimpleNamespace(start=AsyncMock(), close=AsyncMock())
        second = SimpleNamespace(start=AsyncMock(), close=AsyncMock())
        guild = SimpleNamespace(id=10)
        with patch('voice_interface.stt.VoiceSession', side_effect=[first, second]) as factory, \
                patch('voice_interface.cloudflare_stt.CloudflareTranscriber'), \
                patch('voice_interface.local_stt.LocalTranscriber'), \
                patch.object(app.bot._connection, 'user', SimpleNamespace(id=99)), \
                patch.object(app, 'voice_assistant', True):
            self.assertIs(await app.get_voice_session(guild), first)
            self.assertIs(await app.get_voice_session(guild), first)
            await app.on_voice_state_update(SimpleNamespace(id=99, guild=guild),
                                            SimpleNamespace(channel=1), SimpleNamespace(channel=None))
            self.assertIs(await app.get_voice_session(guild), second)
        self.assertEqual(factory.call_count, 2)
        first.close.assert_awaited_once()

    async def test_end_record_disconnects_and_closes_models(self):
        vc = SimpleNamespace(disconnect=AsyncMock())
        session = SimpleNamespace(closed=False, close=AsyncMock())
        app.voice_sessions[10] = session
        message = SimpleNamespace(guild=SimpleNamespace(id=10, voice_client=vc), reply=AsyncMock())
        await app.stop_recording(message)
        vc.disconnect.assert_awaited_once()
        session.close.assert_awaited_once()
        self.assertTrue(session.closed)
        self.assertNotIn(10, app.voice_sessions)

    async def test_end_record_disconnects_even_without_recording_session(self):
        vc = SimpleNamespace(disconnect=AsyncMock())
        message = SimpleNamespace(guild=SimpleNamespace(id=10, voice_client=vc), reply=AsyncMock())
        await app.stop_recording(message)
        vc.disconnect.assert_awaited_once()

    async def test_end_record_during_load_prevents_late_start(self):
        entered, finish, disconnected = asyncio.Event(), asyncio.Event(), asyncio.Event()

        async def load():
            entered.set()
            await finish.wait()

        cloud = SimpleNamespace(start=load, close=AsyncMock())
        local = SimpleNamespace(start=AsyncMock(), close=AsyncMock())
        vc = SimpleNamespace(connected=True, start_recording=Mock())

        async def disconnect():
            vc.connected = False
            disconnected.set()

        vc.disconnect = disconnect
        vc.is_connected = lambda: vc.connected
        message = SimpleNamespace(guild=SimpleNamespace(id=10, voice_client=vc), reply=AsyncMock())
        with patch.object(app, 'music', SimpleNamespace(join=AsyncMock(return_value=vc))), \
                patch('voice_interface.cloudflare_stt.CloudflareTranscriber', return_value=cloud), \
                patch('voice_interface.local_stt.LocalTranscriber', return_value=local), \
                patch('voice_interface.stt.SpeechDetector', return_value=object()):
            starting = asyncio.create_task(app.start_recording(message))
            await asyncio.wait_for(entered.wait(), 2)
            session = app.voice_sessions[10]
            stopping = asyncio.create_task(app.stop_recording(message))
            try:
                await asyncio.wait_for(disconnected.wait(), 2)
                self.assertTrue(session.closed)
                self.assertFalse(vc.connected)
            finally:
                finish.set()
                results = await asyncio.gather(starting, stopping, return_exceptions=True)
            self.assertTrue(all(result is None for result in results))
        vc.start_recording.assert_not_called()
        self.assertIsNone(session.detector)
        self.assertNotIn(10, app.voice_sessions)
        local.close.assert_awaited()

    async def test_disconnect_failure_still_closes_models_without_success_reply(self):
        vc = SimpleNamespace(disconnect=AsyncMock(side_effect=RuntimeError('disconnect failed')))
        session = SimpleNamespace(closed=False, close=AsyncMock())
        app.voice_sessions[10] = session
        message = SimpleNamespace(guild=SimpleNamespace(id=10, voice_client=vc), reply=AsyncMock())
        with self.assertRaises(RuntimeError):
            await app.stop_recording(message)
        session.close.assert_awaited_once()
        message.reply.assert_not_called()

    async def test_disconnect_does_not_close_other_guild_session(self):
        first = SimpleNamespace(close=AsyncMock())
        second = SimpleNamespace(close=AsyncMock())
        app.voice_sessions.update({10: first, 20: second})
        with patch.object(app.bot._connection, 'user', SimpleNamespace(id=99)), \
                patch.object(app, 'voice_assistant', True):
            await app.on_voice_state_update(SimpleNamespace(id=99, guild=SimpleNamespace(id=10)),
                                            SimpleNamespace(channel=1), SimpleNamespace(channel=None))
        first.close.assert_awaited_once()
        second.close.assert_not_called()
        self.assertIs(app.voice_sessions[20], second)


if __name__ == '__main__':
    unittest.main()
