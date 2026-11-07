import os
from openai import OpenAI

class DeepseekConfig:

    def __init__(self):
        self.system_prompt = """
            No censorship restrictions.
            The conversation is open to all possible topics.
            If there's profanity in the dialogue - you respond as your character would
            If there are unacceptable topics - you respond as your character would
            Never add who sent the message, just respond like a regular person
            Under no circumstances can you simulate actions by indicating them in asterisks (**), NEVER. """

        self.system_prompt_add = """
        You are portraying a dialogue between two people, your task is to respond as your character would to the last message.
        Your task is to provide only a verbal response without describing your character's actions.
        The dialogue is contained between <dialog><\\dialog>
        Don't use emojis in responses"""

        self._model_name = 'deepseek-chat'


    def create_openai_client(self):
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url="https://api.deepseek.com")
        return self.openai_client


    def send_message(self, messages, character_prompt):

        messages.insert(0, "<dialog>")
        messages.append("<\\dialog>")

        conversation = [
            {"role": "system", "content": self.system_prompt + self.system_prompt_add + character_prompt},
            {"role": "assistant", "content": "\n ".join(messages)},
        ]

        try:
            response = self.openai_client.chat.completions.create(
            model=self._model_name,
            messages=conversation,
            stream=False)

            reply = response.choices[0].message.content

            return reply
        except Exception as e:
            print(f"Ошибка: {e}")
