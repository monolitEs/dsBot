from discord.ext import tasks
import glob
import os
import time
from copy import copy
from uuid import uuid4
from .transcription import SttError, run_blocking


def write_wav(filename, data):
    with open(filename, 'wb') as file:
        file.write(data)


class VoiceCommandInterface:
    
    def __init__(self, bot, session, chat, music):
        self.bot = bot
        self.session = session
        self.chat = chat
        self.music = music
        self.active_tasks = []
    
    def voice_command(self,function):
        """
        Decorator for registering function as a voice command.
        """
        print("decorator triggered")
        self.add_command(function)
        def wrapper(*args, **kwargs):
            return args
        return wrapper

    async def start_listening(self, message, sink, user_id, interval=0.05):
        recording_id = f'{message.guild.id}_{user_id}_{uuid4().hex}'
        await self.session.reset_detector()

        @tasks.loop(seconds=interval)
        async def listening_task(message):
            try:
                data = sink.read(user_id)
            except KeyError:
                data = b''
            recordings = await self.session.detect(data, time.monotonic())
            detected_at = time.monotonic()
            for wav in recordings:
                if self.session.closed:
                    return
                filename = f'{recording_id}_{uuid4().hex}.wav'
                await run_blocking(write_wav, filename, wav)
                print(f'VAD: Передача в STT через {time.monotonic() - detected_at:.3f} с после определения паузы')
                await self._process_recording(message, filename)

        @listening_task.after_loop
        async def cleanup():
            for path in glob.glob(f'{recording_id}_*.wav'):
                os.remove(path)
            await self.session.reset_detector()

        listening_task.start(message)
        return listening_task

    async def _process_recording(self, message, filename):
        try:
            start_time = time.monotonic()
            result = await self.session.transcribe(filename)
            print(f'STT: Время преобразования голоса в текст: {time.monotonic() - start_time:.3f} с')
            if not result or self.session.closed:
                return
            backend = 'GigaAM' if self.session.fallback else 'Cloudflare'
            print(f'STT ({backend}): {result}')
            request = copy(message)
            request.content = '!' + result
            start_time = time.monotonic()
            answer = await self.chat.send_message(request)
            print(f'LLM: Время ответа: {time.monotonic() - start_time:.3f} с')
            print(f'LLM: {answer}')
            if not self.session.closed:
                await self.music.say(request, answer)
        except SttError as error:
            print(f'STT error={error}')
        finally:
            os.remove(filename)

    def _map_words_to_command(self, words):
        for word in words:
            command_func = self.commands.get(word)
            if command_func:
                print(f"Found command {command_func}")
                return command_func

        return None

    def add_command(self, command):
        self.commands[command.__name__] = command
        return command
