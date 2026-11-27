"""Synchronous playback controls and their async Discord entry points."""
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, call, patch

import dsbot.bot as app
from dsbot.music.player import Music
from dsbot.chat.deepseek import LlmError


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
    async def test_context_command_shows_and_changes_global_budget(self):
        from dsbot.chat.service import Chat
        with patch('dsbot.chat.service.DeepseekConfig.create_openai_client'):
            chat = Chat()
        ctx = SimpleNamespace(respond=AsyncMock())
        with patch.object(app, 'chat', chat):
            await app.context.callback(ctx)
            self.assertIn('256000', ctx.respond.call_args.args[0])
            await app.context.callback(ctx, 8192)
            self.assertEqual(chat.context_tokens, 8192)
            self.assertIn('8192', ctx.respond.call_args.args[0])
            await app.context.callback(ctx, 256000)
            self.assertEqual(chat.context_tokens, 256000)
        option = app.context.options[0]
        self.assertFalse(option.required)
        self.assertEqual((option.min_value, option.max_value), (8192, 256000))

    async def test_failed_chat_replies_with_safe_error_and_stale_result_is_silent(self):
        chat = SimpleNamespace(send_message=AsyncMock(side_effect=LlmError('unavailable')))
        message = SimpleNamespace(author=SimpleNamespace(bot=False), content='!question', reply=AsyncMock())
        with patch.object(app, 'chat', chat):
            await app.on_message(message)
            message.reply.assert_awaited_once_with('unavailable')
            message.reply.reset_mock()
            chat.send_message.side_effect = None
            chat.send_message.return_value = None
            await app.on_message(message)
            message.reply.assert_not_called()

    async def test_bot_shutdown_closes_llm_even_without_discord_login(self):
        chat = SimpleNamespace(close=AsyncMock())
        with patch.object(app, 'chat', chat), \
                patch('discord.ext.commands.Bot.close', new_callable=AsyncMock) as close:
            await app.bot.close()
            chat.close.assert_awaited_once()
            close.assert_awaited_once()
            chat.close.side_effect = RuntimeError('close failed')
            with self.assertRaises(RuntimeError):
                await app.bot.close()
            self.assertEqual(close.await_count, 2)

    async def test_text_message_awaits_chat_result_in_reply(self):
        chat = Mock()
        chat.send_message = AsyncMock(return_value='answer')
        message = SimpleNamespace(author=SimpleNamespace(bot=False), content='!question', reply=AsyncMock())
        with patch.object(app, 'chat', chat):
            await app.on_message(message)
        chat.send_message.assert_awaited_once_with(message)
        message.reply.assert_awaited_once_with('answer')

    async def test_human_text_is_saved_but_bot_text_and_controls_are_excluded(self):
        chat = Mock()
        chat.send_message = AsyncMock()
        human = SimpleNamespace(bot=False)
        note = SimpleNamespace(author=human, content='release Friday')
        with patch.object(app, 'chat', chat), patch.object(app, 'voice_assistant', False):
            await app.on_message(note)
            await app.on_message(SimpleNamespace(author=human, content='!record'))
            await app.on_message(SimpleNamespace(author=SimpleNamespace(bot=True), content='bot output'))
            await app.on_message(SimpleNamespace(author=human, content=''))
        chat.remember_message.assert_called_once_with(note)
        chat.send_message.assert_not_awaited()

    async def test_music_commands_invoke_sync_controls_without_awaiting_result(self):
        music = Mock(spec=Music)
        for command in ('pause', 'resume', 'stop', 'skip'):
            with self.subTest(command=command), patch.object(app, 'music', music):
                music.reset_mock()
                getattr(music, command).return_value = None
                message = SimpleNamespace(author=SimpleNamespace(bot=False), content='!' + command)
                self.assertIsNone(await app.on_message(message))
                self.assertEqual(music.mock_calls, [getattr(call, command)(message)])

    async def test_slash_callbacks_mutate_chat_then_respond(self):
        chat = Mock()
        chat.changecharacter.return_value = chat.stopmessage.return_value = None
        ctx = SimpleNamespace(respond=AsyncMock())
        with patch.object(app, 'chat', chat):
            await app.changecharacter.callback(ctx, 'guide', 'mentor')
            chat.changecharacter.assert_called_once_with('guide', 'mentor')
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
