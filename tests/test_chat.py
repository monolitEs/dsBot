"""Chat history and token-budget contracts, without LLM or tokenizer downloads."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, call, patch

from dsbot.chat.service import Chat


def message(user_id, text, name='user'):
    return SimpleNamespace(author=SimpleNamespace(id=user_id, name=name), content=text)


class TokenBudgetTests(unittest.TestCase):
    def test_count_includes_message_overhead_and_encodes_each_message(self):
        encoder = Mock()
        encoder.encode.side_effect = [[1, 2], [3, 4, 5]]
        with patch('dsbot.chat.service.tiktoken.encoding_for_model', return_value=encoder):
            self.assertEqual(Chat.__new__(Chat).count_tokens(['first', 'second']), 13)
        self.assertEqual(encoder.encode.call_args_list, [call('first'), call('second')])

    def test_unknown_model_falls_back_to_cl100k_encoding(self):
        encoder = Mock()
        encoder.encode.return_value = [7]
        with patch('dsbot.chat.service.tiktoken.encoding_for_model', side_effect=KeyError), \
                patch('dsbot.chat.service.tiktoken.get_encoding', return_value=encoder) as fallback:
            self.assertEqual(Chat.__new__(Chat).count_tokens(['hello']), 5)
            fallback.assert_called_once_with('cl100k_base')
        encoder.encode.assert_called_once_with('hello')

    def test_trim_discards_old_turns_and_keeps_first_message_within_budget(self):
        chat = Chat.__new__(Chat)
        encoder = Mock()
        encoder.encode.side_effect = lambda text: list(text)
        history = ['first', 'old', 'recent', 'last']
        with patch('dsbot.chat.service.tiktoken.encoding_for_model', return_value=encoder):
            self.assertIs(chat.trim_history(history, max_tokens=17), history)
            self.assertEqual(history, ['first', 'last'])
            self.assertEqual(chat.count_tokens(history), 17)
            self.assertEqual(chat.trim_history(history, max_tokens=17), ['first', 'last'])


class ChatHistoryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        config_type = self.enterContext(patch('dsbot.chat.service.DeepseekConfig'))
        self.config = config_type.return_value
        self.chat = Chat()

    def test_history_accumulates_per_user_and_keeps_original_messages(self):
        first = message(10, '!hello', 'alice')
        alice = self.chat.get_recent_messages(first)
        bob = self.chat.get_recent_messages(message(20, '!hi', 'bob'))
        self.assertIs(self.chat.get_recent_messages(message(10, '!again', 'alice')), alice)
        self.assertEqual(alice.messages, ['alice: hello', 'alice: again'])
        self.assertEqual(bob.messages, ['bob: hi'])
        self.assertIsNot(alice, bob)
        self.assertEqual(alice.character_prompt, Chat.CHARACTER_PROMPT)
        self.assertEqual(alice.character_name, Chat.CHARACTER_NAME)
        self.assertEqual(first.content, '!hello')

    def test_persona_changes_for_existing_and_new_user_and_reset_is_isolated(self):
        alice = self.chat.get_recent_messages(message(10, '!hello'))
        bob = self.chat.get_recent_messages(message(20, '!hi'))
        with patch.object(self.chat.log_manager, 'logging') as log:
            self.chat.changecharacter(message(10, ''), 'pirate', 'captain')
            self.chat.changecharacter(message(30, ''), 'guide', 'mentor')
        self.assertEqual(alice.character_prompt, 'Моя роль: pirate')
        self.assertEqual(alice.character_name, 'captain')
        self.assertEqual(self.chat.users_data[30].character_prompt, 'Моя роль: guide')
        self.assertEqual(self.chat.users_data[30].character_name, 'mentor')
        self.assertEqual(log.call_args_list, [call('Моя роль: pirate'), call('Моя роль: guide')])
        self.chat.stopmessage(message(10, ''))
        self.chat.stopmessage(message(999, ''))
        self.assertEqual(alice.messages, [])
        self.assertEqual(bob.messages, ['user: hi'])
        self.assertNotIn(999, self.chat.users_data)

    def test_send_passes_history_snapshot_and_persona_then_stores_named_reply(self):
        self.config.send_message.return_value = 'answer'
        request = message(10, '!question', 'alice')
        self.chat.get_recent_messages(message(10, '!hello', 'alice'))
        self.chat.changecharacter(request, 'guide', 'mentor')
        with patch.object(self.chat, 'count_tokens', return_value=5):
            self.assertEqual(self.chat.send_message(request), 'answer')
        self.config.send_message.assert_called_once_with(
            ['alice: hello', 'alice: question'], 'Моя роль: guide')
        self.assertEqual(self.chat.users_data[10].messages,
                         ['alice: hello', 'alice: question', 'mentor: answer'])
        self.assertEqual(request.content, '!question')
