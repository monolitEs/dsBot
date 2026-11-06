import os
from pathlib import Path

import discord
from discord.ext import commands
from dotenv import load_dotenv
from discord import Option

from modules.role_manager import RoleManager
from modules.ytdl import Music
from modules.chat import Chat

intents = discord.Intents.all()
bot = commands.Bot(command_prefix='!', intents=intents)
chat = None
music = None
role_manager = None
voice_assistant = None
voice_sessions = {}

def init(voice_assistant_bool):
    load_dotenv()

    global chat
    chat = Chat()
    global music
    music = Music(bot, chat)
    global role_manager
    role_manager = RoleManager(bot)

    global voice_assistant
    voice_assistant = voice_assistant_bool

    return bot



# Смена персонажа
@bot.slash_command(name="changecharacter", description="Смена характера бота")
async def changecharacter(ctx, arg: Option(str, description="Описание характера", required=True),
                          arg2: Option(str, description="Имя", required=True)):
    await chat.changecharacter(ctx, arg, arg2)
    await ctx.respond("✅ Системный промпт обновлен! \n Теперь моя роль: " + arg)

# Прекращение контекста для пользователя
@bot.slash_command(name="stopmessage", description="Стоп слово для бота")
async def stopmessage(ctx):
    await chat.stopmessage(ctx)
    await ctx.respond("❎ Контекст очищен")

@bot.event
async def on_ready():
    print(f"Бот {bot.user} готов!")


async def get_voice_session(guild):
    from voice_interface.stt import VoiceSession
    from voice_interface.cloudflare_stt import CloudflareTranscriber
    from voice_interface.local_stt import LocalTranscriber

    if guild.id not in voice_sessions:
        voice_sessions[guild.id] = VoiceSession(
            CloudflareTranscriber(os.getenv('CLOUDFLARE_ACCOUNT_ID', ''), os.getenv('CLOUDFLARE_API_TOKEN', '')),
            LocalTranscriber(Path(os.getenv('GIGAAM_MODEL_DIR', '/data/models/gigaam-ctc'))))
    session = voice_sessions[guild.id]
    try:
        await session.start()
        return session
    except BaseException:
        if voice_sessions.get(guild.id) is session:
            voice_sessions.pop(guild.id)
        await session.close()
        raise


@bot.event
async def on_voice_state_update(member, before, after):
    if not voice_assistant or member.id != bot.user.id:
        return
    if after.channel is None:
        session = voice_sessions.pop(member.guild.id, None)
        if session:
            await session.close()
    elif before.channel is None:
        try:
            await get_voice_session(member.guild)
        except Exception as error:
            print(f'STT startup error={type(error).__name__}')

# Обработка сообщений
@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    if message.content.startswith("!"):

        if message.content.startswith("!play "):
            result = message.content.replace("!play ", "", 1)
            return await music.play(message, url=result)
        elif message.content.startswith("!join"):
            return await music.join(message)
        elif message.content.startswith("!leave"):
            return await music.leave(message)
        elif message.content.startswith("!pause"):
            return await music.pause(message)
        elif message.content.startswith("!resume"):
            return await music.resume(message)
        elif message.content.startswith("!stop"):
            return await music.stop(message)
        elif message.content.startswith("!skip"):
            return await music.skip(message)
        elif message.content.startswith("!queue"):
            return await music.queue(message)
        elif message.content.startswith("!record"):
            if voice_assistant:
                return await start_recording(message)
        elif message.content.startswith("!endrecord"):
            return await stop_recording(message)
        elif message.content.startswith("!role"):
            return await role_manager.create_role(message, "admin", 0, 0, 0)

        return await message.reply(await chat.send_message(message))

async def start_recording(message):
    from voice_interface.sinks import StreamSink
    from voice_interface.core import VoiceCommandInterface

    vc = await music.join(message)
    if vc is None:
        return
    try:
        session = await get_voice_session(message.guild)
    except Exception as error:
        print(f'STT startup error={type(error).__name__}')
        return await message.reply('Не удалось загрузить STT. Проверьте настройки и файлы моделей.')
    async with session.record_lock:
        if session.closed or not vc.is_connected():
            return
        if vc.is_recording():
            return await message.reply('Запись уже идёт!')
        await session.stop()
        sink = StreamSink(filters={'users': [message.author.id]})
        vc.start_recording(sink, recording_finished, None)
        interface = VoiceCommandInterface(bot, session, chat, music)
        session.listener = await interface.start_listening(message, sink, message.author.id)

def recording_finished(sink, _context):
    """PR3159 passes the sink; an extra argument avoids its empty-args bug."""
    error = getattr(sink.client._reader, 'error', None)
    if error is not None:
        print(f'STT receive error={type(error).__name__}')

async def stop_recording(message):
    voice_client = message.guild.voice_client
    session = voice_sessions.get(message.guild.id)
    if not session and not voice_client:
        await message.reply('Бот не подключён к голосовому каналу.')
        return
    if session:
        session.closed = True  # Reject pending starts before the first await.
    try:
        if voice_client:
            await voice_client.disconnect()
    finally:
        if session:
            if voice_sessions.get(message.guild.id) is session:
                voice_sessions.pop(message.guild.id)
            await session.close()
    await message.reply('Запись остановлена, бот вышел из голосового канала.')




