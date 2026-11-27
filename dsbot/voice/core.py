from discord.ext import tasks
import glob
import os
import time
from copy import copy
from uuid import uuid4
from .transcription import SttError, run_blocking
from dsbot.chat.deepseek import LlmError


def write_wav(filename, data):
    with open(filename, 'wb') as file:
        file.write(data)


class VoiceCommandInterface:
    
    def __init__(self, voice_client, session, chat, music):
        self.voice_client = voice_client
        self.session = session
        self.chat = chat
        self.music = music

    def _in_channel(self, channel_id):
        channel = self.voice_client.channel
        return (not self.session.closed and self.voice_client.is_connected()
                and channel is not None and channel.id == channel_id)

    async def start_listening(self, message, sink, user_id, interval=0.05, *,
                              listen_while_speaking=False):
        """Set listen_while_speaking=True to restore reception during TTS."""
        recording_id = f'{message.guild.id}_{user_id}_{uuid4().hex}'
        channel_id = self.voice_client.channel.id
        await self.session.reset_detector()

        @tasks.loop(seconds=interval)
        async def listening_task(message):
            await self._listen_once(
                message, sink, user_id, channel_id, recording_id,
                listen_while_speaking=listen_while_speaking)

        @listening_task.after_loop
        async def cleanup():
            for path in glob.glob(f'{recording_id}_*.wav'):
                os.remove(path)
            await self.session.reset_detector()

        listening_task.start(message)
        return listening_task

    async def _listen_once(self, message, sink, user_id, channel_id, recording_id, *,
                           listen_while_speaking):
        try:
            data = sink.read(user_id)
        except KeyError:
            data = b''
        if not self._in_channel(channel_id):
            return
        recordings = await self.session.detect(data, time.monotonic())
        detected_at = time.monotonic()
        if not listen_while_speaking:
            recordings = recordings[:1]  # No backlog of replies after playback.
        if not self._in_channel(channel_id):
            return
        pending = [(self.chat.reserve_voice(message, channel_id, recording_id), wav)
                   for wav in recordings]
        try:
            for entry, wav in pending:
                if not self._in_channel(channel_id):
                    return
                if entry is None:
                    continue
                filename = f'{recording_id}_{entry.event.sequence}.wav'
                await run_blocking(write_wav, filename, wav)
                print(f'VAD: Передача в STT через {time.monotonic() - detected_at:.3f} с после определения паузы')
                await self._process_recording(
                    message, filename, entry, sink=None if listen_while_speaking else sink)
        finally:
            for entry, _wav in pending:
                if entry is not None:
                    self.chat.skip_voice(entry, 'interrupted')

    async def _process_recording(self, message, filename, entry, *, sink=None):
        channel_id = entry.event.channel_id
        try:
            start_time = time.monotonic()
            result = await self.session.transcribe(filename)
            print(f'STT: Время преобразования голоса в текст: {time.monotonic() - start_time:.3f} с')
            if not self._in_channel(channel_id):
                self.chat.skip_voice(entry, 'channel_unavailable')
                return
            if not result:
                self.chat.skip_voice(entry, 'empty_transcription')
                return
            backend = 'GigaAM' if self.session.fallback else 'Cloudflare'
            print(f'STT ({backend}): {result}')
            request = copy(message)
            request.content = '!' + result
            start_time = time.monotonic()
            answer = await self.chat.answer_voice(entry, result)
            if answer is None:
                return
            print(f'LLM: Время ответа: {time.monotonic() - start_time:.3f} с')
            print(f'LLM: {answer}')
            if self._in_channel(channel_id):
                if sink is None:
                    await self.music.say(request, answer)
                else:
                    sink.set_listening(False)
                    try:
                        await self.session.reset_detector()
                        if self._in_channel(channel_id) and self.chat.can_play_voice(entry):
                            await self.music.say(request, answer, wait=True)
                    finally:
                        try:
                            await self.session.reset_detector()
                        finally:
                            sink.set_listening(True)
        except SttError as error:
            self.chat.skip_voice(entry, 'stt_error')
            print(f'STT error={error}')
        except LlmError:
            print('LLM: Ответ недоступен')
        finally:
            os.remove(filename)
