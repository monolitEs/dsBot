import tiktoken  # Для подсчёта токенов
import asyncio


from dsbot.chat.deepseek import DeepseekConfig, LlmError
from dsbot.logging import LogManager


class Chat:

    DEFAULT_CHARACTER_NAME = "person"
    DEFAULT_CHARACTER_PROMPT = """
        Ассистент
    """

    MAX_TOKENS = 256000
    
    class UserMessages:
        def __init__(self):
            self.messages = []
            self.lock = asyncio.Lock()
            self.version = 0
    
    
    def __init__(self):
        self.log_manager = LogManager()
        self.config = DeepseekConfig()
        self.users_data = {}
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
        if ctx.author.id in self.users_data:
            self.users_data[ctx.author.id].messages = []
            self.users_data[ctx.author.id].version += 1

    def set_context_tokens(self, tokens):
        if not 8192 <= tokens <= self.MAX_TOKENS:
            raise ValueError('Размер контекста должен быть от 8192 до 256000 токенов.')
        self.context_tokens = tokens
        self.version += 1
            
    #Получает все сообщения для данного юзера
    def get_recent_messages(self, message):
        messages = self.UserMessages()
        if message.author.id in self.users_data:
            messages = self.users_data[message.author.id]
        else:
            self.users_data[message.author.id] = messages

        messages.messages.append({'role': 'user', 'content':
            f'[id={message.author.id}, name={message.author.name}] {message.content[1:]}'})

        return messages

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
    
    #Отправка сообщения с общим персонажем бота
    async def send_message(self, message):
        user_messages = self.users_data.setdefault(message.author.id, self.UserMessages())
        async with user_messages.lock:
            self.get_recent_messages(message)
            version = (self.version, user_messages.version)
            persona = f'{self.character_prompt}\nТвоё имя: {self.character_name}'
            conversation = [self.config.system_message(persona),
                            *[item.copy() for item in user_messages.messages]]
            conversation = await asyncio.to_thread(self.trim_history, conversation, self.context_tokens - 4096)
            if version != (self.version, user_messages.version):
                return None
            user_messages.messages = [item.copy() for item in conversation[1:]]
            reply = await self.config.send_message(conversation)
            if version != (self.version, user_messages.version):
                return None
            user_messages.messages.append({'role': 'assistant', 'content': reply})
            return reply

    async def close(self):
        await self.config.close()
