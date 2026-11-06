import tiktoken  # Для подсчёта токенов


from modules.models_configs.deepseek import DeepseekConfig
from utils.log_manager import LogManager


class Chat:

    CHARACTER_NAME = "person"
    CHARACTER_PROMPT = """
        Ассистент
    """

    MAX_TOKENS = 15000
    
    class UserMessages:
        messages = []
        character_prompt = ""
        character_name = ""
    
    
    def __init__(self):
        self.log_manager = LogManager()
        self.config = DeepseekConfig()
        self.users_data = {}
        self.client = self.config.create_openai_client()
    
    #Смена характера бота
    async def changecharacter(self, ctx, character, name):
        if ctx.author.id not in self.users_data:
            messages = self.UserMessages()
            self.users_data[ctx.author.id] = messages
            
        self.users_data[ctx.author.id].character_prompt = "Моя роль: " + character
        self.users_data[ctx.author.id].character_name = name

        self.log_manager.logging(self.users_data[ctx.author.id].character_prompt)
        
    async def stopmessage(self, ctx):
        if ctx.author.id in self.users_data:
            self.users_data[ctx.author.id].messages = []
            
    #Получает все сообщения для данного юзера
    async def get_recent_messages(self, message):
        messages = self.UserMessages()
        if message.author.id in self.users_data:
            messages = self.users_data[message.author.id]
            messages.messages.append(message.author.name + ": " + message.content[1:])
        else:
            messages.messages = [message.author.name + ": " + message.content[1:]]
            messages.character_prompt = self.CHARACTER_PROMPT
            messages.character_name = self.CHARACTER_NAME
            self.users_data[message.author.id] = messages

        return messages

    #Режет сообщения если токенов больше чем контекст
    #Я правда не знаю действительно ли это работат, я столько с ботом не переписывался, сколько конеткст поставил
    def trim_history(self, messages, max_tokens=MAX_TOKENS):
        while self.count_tokens(messages) > max_tokens:
            messages.pop(1)
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
            total_tokens += len(encoding.encode(message))
        return total_tokens
    
    #Отправка сообщения с тем характером, что юзер задал
    async def send_message(self, message):
        user_messages = await self.get_recent_messages(message)
        user_messages.messages = self.trim_history(user_messages.messages)


        reply = self.config.send_message(user_messages.messages.copy(), user_messages.character_prompt)
        user_messages.messages.append(user_messages.character_name + ": " + reply)

        return reply
