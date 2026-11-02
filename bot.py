import os

import discord
from datetime import datetime 
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
voice_command_interface = None
voice_assistant = None
async_task = None

def init(voice_assistant_bool):
    if voice_assistant_bool:
        raise RuntimeError("Распознавание речи временно отключено. Запустите бота с аргументом false.")

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

    vc = await music.join(message)

    sink = StreamSink()

    vc.start_recording(sink, lambda x, y: x, message.channel)

    global async_task
    async_task = await voice_command_interface.start_listening(message, sink, message.author.id)

async def save_current_recording(sink, channel):
    # Копируем текущие данные и сбрасываем sink для новой записи
    audio_data = sink.audio_data.copy()
    sink.audio_data.clear()  # Очищаем, чтобы не дублировать данные

    for user_id, audio in audio_data.items():
        user = await channel.guild.fetch_member(user_id)
        filename = f"{user.name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.wav"
        with open(filename, "wb") as f:
            f.write(audio.file.getbuffer())

async def stop_recording(message):
    voice_client = message.guild.voice_client
    if not voice_client or not voice_client.recording:
        await message.reply("Сейчас нет активной записи!")
        return
    voice_client.stop_recording()

    if async_task:
        async_task.stop()

    await message.reply("Запись остановлена!")




