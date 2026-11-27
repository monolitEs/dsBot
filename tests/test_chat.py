"""Async chat, global persona and bounded request history without network calls."""
import asyncio
from datetime import timezone
from itertools import count
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, call, patch

from dsbot.chat.service import Chat
from dsbot.chat.deepseek import LlmError


message_ids = count(1)


def message(user_id, text, name='user', *, guild=1, channel=2):
    return SimpleNamespace(
        id=next(message_ids), author=SimpleNamespace(id=user_id, name=name), content=text,
        guild=None if guild is None else SimpleNamespace(id=guild),
        channel=SimpleNamespace(id=channel))


class CharacterEncoder:
    def encode(self, text, **kwargs):
        return list(text)


class TokenBudgetTests(unittest.TestCase):
    def setUp(self):
        self.chat = Chat.__new__(Chat)
        self.enterContext(patch('dsbot.chat.service.tiktoken.encoding_for_model',
                                return_value=CharacterEncoder()))

    def test_count_includes_roles_content_and_overhead(self):
        self.assertEqual(self.chat.count_tokens([{'role': 'user', 'content': 'abc'}]), 11)

    def test_unknown_model_uses_fallback_encoding_and_allows_special_token_text(self):
        with patch('dsbot.chat.service.tiktoken.encoding_for_model', side_effect=KeyError), \
                patch('dsbot.chat.service.tiktoken.get_encoding', return_value=CharacterEncoder()) as fallback:
            self.assertGreater(self.chat.count_tokens([
                {'role': 'user', 'content': '<|endoftext|>'}]), 0)
        fallback.assert_called_once_with('cl100k_base')

    def test_trim_removes_old_turn_and_its_answer_preserving_system_and_latest(self):
        system = {'role': 'system', 'content': 'persona'}
        latest = {'role': 'user', 'content': 'new'}
        history = [system, {'role': 'user', 'content': 'old'},
                   {'role': 'assistant', 'content': 'answer'}, latest]
        budget = self.chat.count_tokens([system, latest]) + 5
        self.assertIs(self.chat.trim_history(history, budget), history)
        self.assertEqual(history, [system, latest])
        self.assertEqual(self.chat.trim_history(history, budget), [system, latest])

    def test_single_oversized_user_is_truncated_without_index_error(self):
        history = [{'role': 'system', 'content': 'persona'},
                   {'role': 'user', 'content': 'long message' * 200}]
        self.chat.trim_history(history, 50)
        self.assertLessEqual(self.chat.count_tokens(history), 50)
        self.assertEqual(history[0]['content'], 'persona')
        self.assertTrue(('long message' * 200).startswith(history[1]['content']))
        self.assertTrue(history[1]['content'])

    def test_system_that_leaves_no_user_space_is_controlled_error(self):
        history = [{'role': 'system', 'content': 'persona' * 100},
                   {'role': 'user', 'content': 'question'}]
        with self.assertRaises(LlmError):
            self.chat.trim_history(history, 30)


class ChatHistoryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.enterContext(patch('dsbot.chat.service.DeepseekConfig.create_openai_client'))
        self.enterContext(patch('dsbot.chat.service.tiktoken.encoding_for_model',
                                return_value=CharacterEncoder()))
        self.chat = Chat()
        self.config = self.chat.config
        self.config.send_message = AsyncMock(return_value='answer')
        self.config.close = AsyncMock()

    async def test_two_authors_share_channel_and_duplicate_text_is_ignored(self):
        first = message(10, 'release Friday', 'alice')
        self.chat.remember_message(first)
        self.chat.remember_message(first)
        self.config.send_message.assert_not_awaited()
        question = message(20, '!when is the release?', 'bob')
        await self.chat.send_message(question)
        self.assertIsNone(await self.chat.send_message(question))
        self.config.send_message.assert_awaited_once()
        self.assertEqual(self.config.send_message.call_args.args[0][1:], [
            {'role': 'user', 'content': '[id=10, name=alice] release Friday'},
            {'role': 'user', 'content': '[id=20, name=bob] when is the release?'}])
        self.assertEqual(question.content, '!when is the release?')

    async def test_history_isolated_by_guild_channel_and_dm(self):
        for attrs in ({}, {'channel': 3}, {'guild': 9}, {'guild': None}):
            with self.subTest(attrs=attrs):
                self.chat.remember_message(message(10, str(attrs), **attrs))
                await self.chat.send_message(message(10, '!question', **attrs))
                snapshot = self.config.send_message.call_args.args[0]
                self.assertEqual(len(snapshot), 3)
                self.assertTrue(snapshot[1]['content'].endswith(str(attrs)))

    async def test_global_persona_is_used_by_other_users_and_reset_is_isolated(self):
        await self.chat.send_message(message(10, '!hello'))
        self.chat.remember_message(message(20, 'hi', channel=3))
        with patch.object(self.chat.log_manager, 'logging') as log:
            self.chat.changecharacter('pirate', 'captain')
            self.chat.changecharacter('guide', 'mentor')
        self.assertEqual(log.call_args_list, [call('Моя роль: pirate'), call('Моя роль: guide')])
        await self.chat.send_message(message(20, '!question', channel=3))
        snapshot = self.config.send_message.call_args.args[0]
        self.assertIn('Моя роль: guide', snapshot[0]['content'])
        self.assertIn('mentor', snapshot[0]['content'])
        self.chat.stopmessage(message(20, ''))
        self.chat.stopmessage(message(999, '', channel=99))
        self.assertFalse(self.chat.histories[1, 2].entries)
        self.assertTrue(self.chat.histories[1, 3].entries)
        self.assertNotIn((1, 99), self.chat.histories)
        self.assertEqual(self.chat.character_name, 'mentor')

    async def test_send_uses_detached_snapshot_and_real_roles(self):
        request = message(10, '!question', 'alice')
        self.chat.remember_message(message(10, 'hello', 'alice'))
        self.assertEqual(await self.chat.send_message(request), 'answer')
        snapshot = self.config.send_message.call_args.args[0]
        self.assertEqual([item['role'] for item in snapshot], ['system', 'user', 'user'])
        entry = self.chat.histories[1, 2].entries[f'text:{request.id}']
        self.assertEqual(entry.reply, 'answer')
        snapshot[-1]['content'] = 'changed outside'
        self.assertEqual(entry.content, 'question')
        self.assertEqual(request.content, '!question')

    async def test_slow_llm_allows_other_channel_and_orders_same_channel(self):
        entered, release = asyncio.Event(), asyncio.Event()
        snapshots = []

        async def answer(conversation):
            snapshots.append(conversation)
            if 'first' in conversation[-1]['content']:
                entered.set()
                await release.wait()
            return 'reply ' + conversation[-1]['content']

        self.config.send_message.side_effect = answer
        first = asyncio.create_task(self.chat.send_message(message(10, '!first')))
        await asyncio.wait_for(entered.wait(), 2)
        self.chat.remember_message(message(20, 'background'))
        second = asyncio.create_task(self.chat.send_message(message(20, '!second')))
        try:
            # This await itself is a scheduler heartbeat; it completes while first waits.
            other = await asyncio.wait_for(self.chat.send_message(message(10, '!other', channel=3)), 2)
            self.assertIn('other', other)
            self.assertFalse(first.done())
            self.assertFalse(second.done())
            self.assertEqual(len(snapshots), 2)
        finally:
            release.set()
            await asyncio.gather(first, second)
        self.assertEqual([item['role'] for item in snapshots[-1]],
                         ['system', 'user', 'assistant', 'user', 'user'])
        self.assertIn('first', snapshots[-1][2]['content'])
        self.assertIn('background', snapshots[-1][3]['content'])
        self.assertEqual(len(snapshots[0]), 2)

    async def test_waiting_question_keeps_context_evicted_by_later_message(self):
        with patch('dsbot.chat.history.ConversationHistory.MAX_EVENTS', 3):
            fact = message(10, 'release on Friday')
            self.chat.remember_message(fact)
            earlier = self.chat.reserve_voice(message(10, ''), 2, 'session')
            question = asyncio.create_task(self.chat.send_message(message(20, '!when is release?')))
            try:
                await asyncio.sleep(0)
                self.config.send_message.assert_not_awaited()
                self.chat.remember_message(message(30, 'later message'))
                self.assertNotIn(f'text:{fact.id}', self.chat.histories[1, 2].entries)
                await self.chat.answer_voice(earlier, 'earlier question')
                self.assertEqual(await asyncio.wait_for(question, 2), 'answer')
            finally:
                if not question.done():
                    question.cancel()
                    await asyncio.gather(question, return_exceptions=True)
            messages = self.config.send_message.await_args.args[0][1:]
            self.assertEqual(messages, [
                {'role': 'user', 'content': '[id=10, name=user] release on Friday'},
                {'role': 'user', 'content': '[id=10, name=user] earlier question'},
                {'role': 'assistant', 'content': 'answer'},
                {'role': 'user', 'content': '[id=20, name=user] when is release?'},
            ])

    async def test_voice_question_keeps_context_evicted_during_transcription(self):
        with patch('dsbot.chat.history.ConversationHistory.MAX_EVENTS', 2):
            fact = message(10, 'release on Friday', channel=9)
            self.chat.remember_message(fact)
            voice = self.chat.reserve_voice(message(20, ''), 9, 'session')
            self.chat.remember_message(message(30, 'later message', channel=9))
            self.assertNotIn(f'text:{fact.id}', self.chat.histories[1, 9].entries)

            self.assertEqual(await self.chat.answer_voice(voice, 'when is release?'), 'answer')
            self.assertEqual(self.config.send_message.await_args.args[0][1:], [
                {'role': 'user', 'content': '[id=10, name=user] release on Friday'},
                {'role': 'user', 'content': '[id=20, name=user] when is release?'},
            ])

    async def test_voice_completion_order_follows_reservations(self):
        alice = self.chat.reserve_voice(message(10, '', 'alice'), 9, 'session')
        bob = self.chat.reserve_voice(message(20, '', 'bob'), 9, 'session')
        self.assertEqual((alice.event.guild_id, alice.event.channel_id, alice.event.source_channel_id),
                         (1, 9, 2))
        self.assertEqual((alice.event.source, alice.event.session_id, alice.event.event_id),
                         ('voice', 'session', 'voice:session:1'))
        self.assertEqual(alice.event.captured_at.tzinfo, timezone.utc)
        later = asyncio.create_task(self.chat.answer_voice(bob, 'question two'))
        try:
            await asyncio.sleep(0)
            self.config.send_message.assert_not_awaited()
            await self.chat.answer_voice(alice, 'question one')
            await asyncio.wait_for(later, 2)
        finally:
            if not later.done():
                later.cancel()
                await asyncio.gather(later, return_exceptions=True)
        first, second = [call.args[0][1:] for call in self.config.send_message.await_args_list]
        self.assertEqual(first, [{'role': 'user', 'content': '[id=10, name=alice] question one'}])
        self.assertEqual(second, first + [{'role': 'assistant', 'content': 'answer'},
                         {'role': 'user', 'content': '[id=20, name=bob] question two'}])
        self.assertIsNone(await self.chat.answer_voice(bob, 'duplicate'))
        self.assertEqual(self.config.send_message.await_count, 2)

    async def test_skipped_voice_releases_later_question_in_same_channel(self):
        missing = self.chat.reserve_voice(message(10, ''), 9, 'session')
        pending = asyncio.create_task(self.chat.send_message(message(20, '!next', channel=9)))
        try:
            await asyncio.sleep(0)
            self.config.send_message.assert_not_awaited()
            self.chat.skip_voice(missing, 'stt_error')
            self.assertEqual(await asyncio.wait_for(pending, 2), 'answer')
        finally:
            if not pending.done():
                pending.cancel()
                await asyncio.gather(pending, return_exceptions=True)
        self.assertEqual(len(self.config.send_message.call_args.args[0]), 2)
        self.assertIsNone(await self.chat.answer_voice(missing, 'late'))

    async def test_history_evicts_oldest_settled_entry_and_keeps_pending_entries(self):
        with patch('dsbot.chat.history.ConversationHistory.MAX_EVENTS', 2):
            pending = self.chat.reserve_voice(message(10, ''), 2, 'session')
            first = message(20, 'first')
            latest = message(20, 'latest')
            self.chat.remember_message(first)
            self.chat.remember_message(latest)
            history = self.chat.histories[1, 2]
            self.assertEqual(list(history.entries), [pending.event.event_id, f'text:{latest.id}'])
            second = self.chat.reserve_voice(message(10, ''), 2, 'session')
            self.assertIsNone(self.chat.reserve_voice(message(10, ''), 2, 'session'))
            self.assertEqual(list(history.entries), [pending.event.event_id, second.event.event_id])

    async def test_reset_clears_waiting_voice_and_text_without_late_reply(self):
        voice = self.chat.reserve_voice(message(10, ''), 2, 'session')
        waiting = asyncio.create_task(self.chat.send_message(message(20, '!old')))
        try:
            await asyncio.sleep(0)
            self.chat.stopmessage(message(30, ''))
            self.assertIsNone(await asyncio.wait_for(waiting, 2))
        finally:
            if not waiting.done():
                waiting.cancel()
                await asyncio.gather(waiting, return_exceptions=True)
        self.assertIsNone(await self.chat.answer_voice(voice, 'late'))
        self.config.send_message.assert_not_awaited()
        await self.chat.send_message(message(30, '!new'))
        self.assertEqual(len(self.config.send_message.call_args.args[0]), 2)

    async def test_reset_persona_and_budget_changes_discard_pending_answers(self):
        for action in ('reset', 'persona', 'budget'):
            with self.subTest(action=action):
                entered, release = asyncio.Event(), asyncio.Event()

                async def answer(conversation):
                    entered.set()
                    await release.wait()
                    return 'stale'

                self.config.send_message.side_effect = answer
                pending = asyncio.create_task(self.chat.send_message(message(10, '!question')))
                await asyncio.wait_for(entered.wait(), 2)
                if action == 'reset':
                    self.chat.stopmessage(message(10, ''))
                elif action == 'persona':
                    self.chat.changecharacter('new', 'new')
                else:
                    self.chat.set_context_tokens(8192)
                release.set()
                self.assertIsNone(await pending)
                entries = self.chat.histories[1, 2].entries
                self.assertTrue(all(entry.reply is None for entry in entries.values()))
                if action == 'reset':
                    self.assertFalse(entries)

    async def test_failure_and_cancellation_add_no_assistant_and_unlock_next_request(self):
        self.config.send_message.side_effect = LlmError('unavailable')
        request = message(10, '!failed')
        with self.assertRaises(LlmError):
            await self.chat.send_message(request)
        entered = asyncio.Event()

        async def wait(conversation):
            entered.set()
            await asyncio.Future()

        self.config.send_message.side_effect = wait
        pending = asyncio.create_task(self.chat.send_message(message(10, '!cancel')))
        await asyncio.wait_for(entered.wait(), 2)
        pending.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await pending
        self.assertTrue(all(entry.reply is None for entry in self.chat.histories[1, 2].entries.values()))
        self.config.send_message.side_effect = None
        self.assertEqual(await asyncio.wait_for(self.chat.send_message(message(10, '!again')), 2), 'answer')

    async def test_reset_suppresses_error_from_old_request(self):
        entered, release = asyncio.Event(), asyncio.Event()

        async def fail(_conversation):
            entered.set()
            await release.wait()
            raise LlmError('late error')

        self.config.send_message.side_effect = fail
        pending = asyncio.create_task(self.chat.send_message(message(10, '!old')))
        try:
            await asyncio.wait_for(entered.wait(), 2)
            self.chat.stopmessage(message(20, ''))
        finally:
            release.set()
            result = await pending
        self.assertIsNone(result)

    async def test_reset_while_tokenizing_does_not_restore_history_or_start_llm(self):
        entered, release = threading.Event(), threading.Event()
        original = self.chat.trim_history

        def trim(*args):
            entered.set()
            if not release.wait(3):
                raise RuntimeError('test timeout')
            return original(*args)

        with patch.object(self.chat, 'trim_history', side_effect=trim):
            pending = asyncio.create_task(self.chat.send_message(message(10, '!question')))
            try:
                self.assertTrue(await asyncio.to_thread(entered.wait, 2))
                self.chat.stopmessage(message(10, ''))
            finally:
                release.set()
                result = await pending
        self.assertIsNone(result)
        self.config.send_message.assert_not_called()
        self.assertFalse(self.chat.histories[1, 2].entries)

    async def test_context_default_range_and_reduction_apply_to_next_request(self):
        self.assertEqual(self.chat.context_tokens, 256000)
        for value in (0, 8191, 256001):
            with self.assertRaises(ValueError):
                self.chat.set_context_tokens(value)
        self.assertEqual(self.chat.context_tokens, 256000)
        await self.chat.send_message(message(10, '!' + 'x' * 10000))
        self.chat.set_context_tokens(8192)
        await self.chat.send_message(message(10, '!new'))
        snapshot = self.config.send_message.call_args.args[0]
        self.assertLessEqual(self.chat.count_tokens(snapshot) + 4096, 8192)
        self.assertEqual([item['role'] for item in snapshot], ['system', 'user'])
        self.assertTrue(snapshot[-1]['content'].endswith('new'))
        self.chat.set_context_tokens(256000)
        await self.chat.close()
        self.config.close.assert_awaited_once()
