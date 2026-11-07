"""Synchronous playback controls and their async Discord entry points."""
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, call, patch

import dsbot.bot as app
from dsbot.music.player import Music


class MusicControlsTests(unittest.TestCase):
    def setUp(self):
        self.music = Music(SimpleNamespace(voice_clients=[]), None)
        self.voice = Mock()
        self.message = SimpleNamespace(guild=SimpleNamespace(id=10), author=object())
        self.enterContext(patch('discord.utils.get', return_value=self.voice))

    def test_pause_only_affects_playing_audio(self):
        for playing in (False, True):
            with self.subTest(playing=playing):
                self.voice.reset_mock()
                self.voice.is_playing.return_value = playing
                self.assertIsNone(self.music.pause(self.message))
                self.assertEqual(self.voice.pause.call_count, int(playing))

    def test_resume_only_affects_paused_audio(self):
        for paused in (False, True):
            with self.subTest(paused=paused):
                self.voice.reset_mock()
                self.voice.is_paused.return_value = paused
                self.assertIsNone(self.music.resume(self.message))
                self.assertEqual(self.voice.resume.call_count, int(paused))

    def test_stop_clears_only_active_guild_queue_and_keeps_idle_queue(self):
        for playing, paused in [(False, False), (True, False), (False, True)]:
            with self.subTest(playing=playing, paused=paused):
                self.voice.reset_mock()
                self.voice.is_playing.return_value = playing
                self.voice.is_paused.return_value = paused
                self.music.queues = {10: ['pending'], 20: ['other']}
                self.assertIsNone(self.music.stop(self.message))
                self.assertEqual(self.voice.stop.call_count, int(playing or paused))
                self.assertEqual(self.music.queues[10], [] if playing or paused else ['pending'])
                self.assertEqual(self.music.queues[20], ['other'])

    def test_skip_stops_before_advancing_queue_and_does_nothing_when_idle(self):
        events = Mock()
        events.attach_mock(self.voice.stop, 'stop')
        self.music.check_queue = Mock()
        events.attach_mock(self.music.check_queue, 'advance')
        for playing in (False, True):
            with self.subTest(playing=playing):
                events.reset_mock()
                self.voice.is_playing.return_value = playing
                self.assertIsNone(self.music.skip(self.message))
                expected = [call.stop(), call.advance(self.message.author, 10)] if playing else []
                self.assertEqual(events.mock_calls, expected)


class DiscordEntryTests(unittest.IsolatedAsyncioTestCase):
    async def test_text_message_uses_sync_chat_result_in_reply(self):
        chat = Mock()
        chat.send_message.return_value = 'answer'
        message = SimpleNamespace(author=object(), content='!question', reply=AsyncMock())
        with patch.object(app, 'chat', chat):
            await app.on_message(message)
        chat.send_message.assert_called_once_with(message)
        message.reply.assert_awaited_once_with('answer')

    async def test_music_commands_invoke_sync_controls_without_awaiting_result(self):
        music = Mock(spec=Music)
        for command in ('pause', 'resume', 'stop', 'skip'):
            with self.subTest(command=command), patch.object(app, 'music', music):
                music.reset_mock()
                getattr(music, command).return_value = None
                message = SimpleNamespace(author=object(), content='!' + command)
                self.assertIsNone(await app.on_message(message))
                self.assertEqual(music.mock_calls, [getattr(call, command)(message)])

    async def test_slash_callbacks_mutate_chat_then_respond(self):
        chat = Mock()
        chat.changecharacter.return_value = chat.stopmessage.return_value = None
        ctx = SimpleNamespace(respond=AsyncMock())
        with patch.object(app, 'chat', chat):
            await app.changecharacter.callback(ctx, 'guide', 'mentor')
            chat.changecharacter.assert_called_once_with(ctx, 'guide', 'mentor')
            ctx.respond.assert_awaited_once_with(
                '✅ Системный промпт обновлен! \n Теперь моя роль: guide')
            ctx.respond.reset_mock()
            await app.stopmessage.callback(ctx)
            chat.stopmessage.assert_called_once_with(ctx)
            ctx.respond.assert_awaited_once_with('❎ Контекст очищен')

    async def test_missing_voice_reply_is_unchanged_for_join_play_and_say(self):
        music = Music(None, None)
        message = SimpleNamespace(author=SimpleNamespace(voice=None), reply=AsyncMock())
        for method, args in [(music.join, ()), (music.play, ('url',)), (music.say, ('text',))]:
            with self.subTest(method=method.__name__):
                message.reply.reset_mock()
                await method(message, *args)
                message.reply.assert_awaited_once_with('Вы не в голосовом канале!')
