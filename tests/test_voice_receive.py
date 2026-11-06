"""Pinned Pycord receive API tests, without Discord sockets or model loading."""
import asyncio
from contextlib import redirect_stdout
import importlib.util
import inspect
import io
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

AVAILABLE = importlib.util.find_spec('discord') is not None
if AVAILABLE:
    import bot as app
    from discord import Object, opus
    from discord.utils import MISSING
    from discord.voice import VoiceClient, VoiceData
    from discord.voice.receive.reader import AudioReader
    from voice_interface.sinks import StreamSink


def voice_client():
    """Real SDK object; only its Discord connection and network workers are fake."""
    client = VoiceClient.__new__(VoiceClient)
    sender = Object(id=42)
    guild = SimpleNamespace(get_member=Mock(return_value=sender))
    client.channel = SimpleNamespace(guild=guild)
    client.client = SimpleNamespace(get_user=Mock(return_value=sender))
    client._connection = SimpleNamespace(mode='xsalsa20_poly1305', secret_key=bytes(32),
        dave_session=None, add_socket_listener=Mock(), remove_socket_listener=Mock())
    client._reader = MISSING
    client._player = client._player_future = None
    client._ssrc_to_id = {101: 42}
    client.is_connected = lambda: True
    return client


@unittest.skipUnless(AVAILABLE, 'Requires bot runtime dependencies')
class VoiceReceiveTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_client_first_record_repeated_record_and_restart(self):
        client = voice_client()
        session = SimpleNamespace(record_lock=asyncio.Lock(), closed=False,
                                  stop=AsyncMock(), listener=None)
        listener = object()
        message = SimpleNamespace(guild=SimpleNamespace(id=10, voice_client=client),
                                  author=SimpleNamespace(id=42), channel=object(), reply=AsyncMock())
        # Execute real VoiceClient/AudioReader startup. Prevent network worker threads only.
        with patch('threading.Thread.start'), \
                patch.object(app, 'music', SimpleNamespace(join=AsyncMock(return_value=client))), \
                patch.object(app, 'get_voice_session', AsyncMock(return_value=session)), \
                patch.object(app, 'recording_finished', Mock(wraps=app.recording_finished)) as finished, \
                patch('voice_interface.core.VoiceCommandInterface.start_listening',
                      AsyncMock(return_value=listener)) as listen:
            try:
                self.assertFalse(hasattr(client, 'recording'))
                self.assertFalse(client.is_recording())
                await app.start_recording(message)
                self.assertTrue(client.is_recording())
                first_sink = client._reader.sink
                self.assertIs(first_sink.client, client)
                self.assertFalse(inspect.iscoroutinefunction(client._reader.after))
                self.assertIs(session.listener, listener)
                await app.start_recording(message)
                listen.assert_awaited_once()
                message.reply.assert_awaited_once_with('Запись уже идёт!')
                client.stop()
                finished.assert_called_once_with(first_sink, None)
                self.assertFalse(client.is_recording())
                await app.start_recording(message)
                self.assertTrue(client.is_recording())
                self.assertIsNot(client._reader.sink, first_sink)
                second_sink = client._reader.sink
                self.assertEqual(listen.await_count, 2)
            finally:
                client.stop()
        self.assertFalse(client.is_recording())
        self.assertEqual(finished.call_count, 2)
        finished.assert_called_with(second_sink, None)
        self.assertEqual(client._connection.add_socket_listener.call_count, 2)
        self.assertEqual(client._connection.remove_socket_listener.call_count, 2)

    async def test_real_opus_decoder_and_router_feed_incremental_pcm(self):
        client = voice_client()
        sink = StreamSink()
        sink.init(client)
        reader = AudioReader(sink, client)
        decoder = reader.packet_router.get_decoder(101)
        pcm = b'\0' * opus.Encoder.FRAME_SIZE
        encoded = opus.Encoder().encode(pcm, opus.Encoder.SAMPLES_PER_FRAME)
        packet = SimpleNamespace(sequence=1, timestamp=960, decrypted_data=encoded)
        data = decoder._process_packet(packet)
        self.assertIsInstance(data, VoiceData)
        self.assertEqual(len(data.pcm), len(pcm))
        self.assertEqual(data.source.id, 42)
        router = reader.packet_router
        decoder.pop_data = Mock(return_value=data)
        router.waiter.register(decoder)
        original_write = sink.write

        def write_once(received, sender):
            original_write(received, sender)
            router._end_thread.set()

        with patch.object(sink, 'write', side_effect=write_once):
            router._do_run()
        self.assertEqual(sink.read(42), data.pcm)
        self.assertEqual(sink.read(42), b'')
        sink.write(data, data.source)
        self.assertEqual(sink.read(42), data.pcm)
        self.assertEqual(sink.read(42, full=True), data.pcm * 2)

    async def test_filter_uses_numeric_user_id_and_unknown_sender_is_ignored(self):
        sink = StreamSink(filters={'users': [42]})
        sender = Object(id=42)
        other = Object(id=77)
        sink.write(VoiceData(None, sender, pcm=b'allowed'), sender)
        sink.write(VoiceData(None, other, pcm=b'filtered'), other)
        sink.write(VoiceData(None, None, pcm=b'unknown'), None)
        self.assertEqual(sink.read(42), b'allowed')
        with self.assertRaises(KeyError):
            sink.read(77)
        with self.assertRaises(KeyError):
            sink.read(None)

    async def test_reader_callback_is_sync_and_reports_only_error_type(self):
        output = io.StringIO()
        self.assertFalse(inspect.iscoroutinefunction(app.recording_finished))
        with redirect_stdout(output):
            sink = SimpleNamespace(client=SimpleNamespace(_reader=SimpleNamespace(error=None)))
            app.recording_finished(sink, None)
            self.assertEqual(output.getvalue(), '')
            sink.client._reader.error = RuntimeError('fake-secret')
            app.recording_finished(sink, None)
        self.assertIn('STT receive error=RuntimeError', output.getvalue())
        self.assertNotIn('fake-secret', output.getvalue())

    async def test_receive_counts_frames_without_logging(self):
        sink = StreamSink()
        sender = Object(id=42)
        payload = b'fake-secret'
        output = io.StringIO()
        with redirect_stdout(output):
            sink.write(VoiceData(None, None, pcm=payload), None)
            for _ in range(249):
                sink.write(VoiceData(None, sender, pcm=payload), sender)
        self.assertEqual(sink.received_frames, 250)
        self.assertEqual(sink.unassigned_frames, 1)
        self.assertEqual(sink.read(42), payload * 249)
        self.assertEqual(output.getvalue(), '')
