"""Async chat, global persona and bounded request history without network calls."""
import asyncio
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, call, patch

from dsbot.chat.service import Chat
from dsbot.chat.deepseek import LlmError


def message(user_id, text, name='user'):
    return SimpleNamespace(author=SimpleNamespace(id=user_id, name=name), content=text)


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

    def test_authors_are_structured_and_original_message_is_unchanged(self):
        first = message(10, '!hello', 'alice')
        alice = self.chat.get_recent_messages(first)
        bob = self.chat.get_recent_messages(message(20, '!hi', 'bob'))
        self.assertIs(self.chat.get_recent_messages(message(10, '!again', 'alice')), alice)
        self.assertEqual(alice.messages, [
            {'role': 'user', 'content': '[id=10, name=alice] hello'},
            {'role': 'user', 'content': '[id=10, name=alice] again'}])
        self.assertEqual(bob.messages, [{'role': 'user', 'content': '[id=20, name=bob] hi'}])
        self.assertIsNot(alice.messages, bob.messages)
        self.assertEqual(first.content, '!hello')

    async def test_global_persona_is_used_by_other_users_and_reset_is_isolated(self):
        await self.chat.send_message(message(10, '!hello'))
        bob = self.chat.get_recent_messages(message(20, '!hi'))
        with patch.object(self.chat.log_manager, 'logging') as log:
            self.chat.changecharacter('pirate', 'captain')
            self.chat.changecharacter('guide', 'mentor')
        self.assertEqual(log.call_args_list, [call('Моя роль: pirate'), call('Моя роль: guide')])
        await self.chat.send_message(message(20, '!question'))
        snapshot = self.config.send_message.call_args.args[0]
        self.assertIn('Моя роль: guide', snapshot[0]['content'])
        self.assertIn('mentor', snapshot[0]['content'])
        self.chat.stopmessage(message(10, ''))
        self.chat.stopmessage(message(999, ''))
        self.assertEqual(self.chat.users_data[10].messages, [])
        self.assertTrue(bob.messages)
        self.assertNotIn(999, self.chat.users_data)
        self.assertEqual(self.chat.character_name, 'mentor')

    async def test_send_uses_detached_snapshot_and_real_roles(self):
        request = message(10, '!question', 'alice')
        self.chat.get_recent_messages(message(10, '!hello', 'alice'))
        self.assertEqual(await self.chat.send_message(request), 'answer')
        snapshot = self.config.send_message.call_args.args[0]
        self.assertEqual([item['role'] for item in snapshot], ['system', 'user', 'user'])
        history = self.chat.users_data[10].messages
        self.assertEqual(history[-1], {'role': 'assistant', 'content': 'answer'})
        snapshot[-1]['content'] = 'changed outside'
        self.assertEqual(history[-2]['content'], '[id=10, name=alice] question')
        self.assertEqual(request.content, '!question')

    async def test_slow_llm_allows_other_user_and_serializes_same_user(self):
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
        second = asyncio.create_task(self.chat.send_message(message(10, '!second')))
        try:
            # This await itself is a scheduler heartbeat; it completes while first waits.
            other = await asyncio.wait_for(self.chat.send_message(message(20, '!other')), 2)
            self.assertIn('other', other)
            self.assertFalse(first.done())
            self.assertFalse(second.done())
            self.assertEqual(len(snapshots), 2)
        finally:
            release.set()
            await asyncio.gather(first, second)
        self.assertEqual([item['role'] for item in snapshots[-1]],
                         ['system', 'user', 'assistant', 'user'])
        self.assertIn('first', snapshots[-1][-2]['content'])

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
                self.assertNotIn({'role': 'assistant', 'content': 'stale'},
                                 self.chat.users_data[10].messages)
                if action == 'reset':
                    self.assertEqual(self.chat.users_data[10].messages, [])

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
        self.assertTrue(all(item['role'] == 'user' for item in self.chat.users_data[10].messages))
        self.config.send_message.side_effect = None
        self.assertEqual(await asyncio.wait_for(self.chat.send_message(message(10, '!again')), 2), 'answer')

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
        self.assertEqual(self.chat.users_data[10].messages, [])

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
