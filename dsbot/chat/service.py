import tiktoken  # Для подсчёта токенов
import asyncio


from dsbot.chat.deepseek import DeepseekConfig, LlmError
from dsbot.chat.history import ConversationHistory
from dsbot.logging import LogManager


class Chat:

    DEFAULT_CHARACTER_NAME = "person"
    DEFAULT_CHARACTER_PROMPT = """
        Ассистент
    """

    MAX_TOKENS = 256000
    
    def __init__(self):
        self.log_manager = LogManager()
        self.config = DeepseekConfig()
        self.histories = {}
        self.client = self.config.create_openai_client()
        self.character_prompt = self.DEFAULT_CHARACTER_PROMPT
        self.character_name = self.DEFAULT_CHARACTER_NAME
        self.context_tokens, self.version = self.MAX_TOKENS, 0
    
    #Смена характера бота
    def changecharacter(self, character, name):
        self.character_prompt = "Моя роль: " + character
        self.character_name = name
        self.version += 1

        self.log_manager.logging(self.character_prompt)
        
    def stopmessage(self, ctx):
        key = (ctx.guild.id if ctx.guild else None, ctx.channel.id)
        if key in self.histories:
            self.histories[key].clear()

    def set_context_tokens(self, tokens):
        if not 8192 <= tokens <= self.MAX_TOKENS:
            raise ValueError('Размер контекста должен быть от 8192 до 256000 токенов.')
        self.context_tokens = tokens
        self.version += 1
            
    def _reserve(self, message, channel_id, session_id=None):
        key = (message.guild.id if message.guild else None, channel_id)
        if key not in self.histories:
            self.histories[key] = ConversationHistory(key)
        history = self.histories[key]
        return history.reserve(message, session_id, self.version)

    def remember_message(self, message):
        entry = self._reserve(message, message.channel.id)
        if entry is not None:
            entry.content = message.content
            entry.settled.set()

    def reserve_voice(self, message, channel_id, session_id):
        return self._reserve(message, channel_id, session_id)

    def skip_voice(self, entry, reason):
        if not entry.settled.is_set():
            print(f'Chat: voice_skipped event={entry.event.event_id} reason={reason}')
            entry.preceding = None
            entry.settled.set()

    #Режет сообщения если токенов больше чем контекст
    def trim_history(self, messages, max_tokens=MAX_TOKENS):
        # Work on a request snapshot: system is kept, oldest turns leave first.
        costs = [self.count_tokens([item]) for item in messages]
        total = sum(costs)
        while total > max_tokens and len(messages) > 2:
            total -= costs.pop(1)
            messages.pop(1)
        while len(messages) > 2 and messages[1]['role'] == 'assistant':
            total -= costs.pop(1)
            messages.pop(1)
        if len(messages) == 2 and total > max_tokens:
            content = messages[-1]['content']
            low, high = 0, len(content)
            while low < high:
                mid = (low + high + 1) // 2
                messages[-1]['content'] = content[:mid]
                if self.count_tokens(messages) <= max_tokens:
                    low = mid
                else:
                    high = mid - 1
            messages[-1]['content'] = content[:low]
        if self.count_tokens(messages) > max_tokens or not messages[-1]['content']:
            raise LlmError('Персонаж слишком длинный для выбранного контекста.')
        return messages

    #Собственно сам подсчет токенов
    def count_tokens(self, messages):
        try:
            encoding = tiktoken.encoding_for_model("cl100k_base")
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")

        tokens_per_message = 4
        total_tokens = 0
        for message in messages:
            total_tokens += tokens_per_message
            total_tokens += len(encoding.encode(message['role']))
            total_tokens += len(encoding.encode(message['content'], disallowed_special=()))
        return total_tokens
    
    async def send_message(self, message):
        entry = self._reserve(message, message.channel.id)
        if entry is not None:
            return await self._answer(entry, message.content[1:])

    async def answer_voice(self, entry, text):
        if entry.content is None and not entry.settled.is_set():
            return await self._answer(entry, text)

    def can_play_voice(self, entry):
        event = entry.event
        history = self.histories[(event.guild_id, event.channel_id)]
        return event.sequence > history.cleared_through and entry.settings_version == self.version

    def _is_current(self, entry):
        return not entry.settled.is_set() and entry.settings_version == self.version

    async def _answer(self, entry, text):
        entry.content = text
        event = entry.event
        history = self.histories[(event.guild_id, event.channel_id)]
        try:
            messages = await history.snapshot(entry)
            if not self._is_current(entry):
                return None
            persona = f'{self.character_prompt}\nТвоё имя: {self.character_name}'
            conversation = [self.config.system_message(persona), *messages]
            conversation = await asyncio.to_thread(self.trim_history, conversation, self.context_tokens - 4096)
            if not self._is_current(entry):
                return None
            reply = await self.config.send_message(conversation)
            if not self._is_current(entry):
                return None
            entry.reply = reply
            return reply
        except LlmError:
            if self._is_current(entry):
                raise
        finally:
            entry.settled.set()

    async def close(self):
        await self.config.close()
