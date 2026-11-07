# dsBot

Discord-бот с текстовым диалогом, музыкой и голосовыми запросами.

## Структура

```text
dsbot/
  bot.py          # Инициализация, события и команды Discord
  roles.py        # Управление ролями Discord
  logging.py      # Вывод сообщений в stdout или файл
  chat/
    service.py    # Контекст диалога и персонаж пользователя
    deepseek.py   # Обращения к DeepSeek
  music/
    player.py     # Музыкальная очередь, yt-dlp и воспроизведение
    tts.py        # Потоковая озвучка gTTS → FFmpeg → PCM
  voice/
    core.py       # Обработка голосового запроса и ответа
    sinks.py      # Приём и буферизация PCM от Discord
    stt.py        # Жизненный цикл голосовой сессии и fallback
    vad.py        # Определение границ речи с Silero
    transcription.py  # Контракт STT и выполнение блокирующих операций
    cloudflare_stt.py # Облачное распознавание
    local_stt.py      # Локальное распознавание GigaAM
tests/            # Offline-проверки приложения и pinned SDK
models/           # Модель Silero VAD
docker/           # Entrypoint, настройки и vendored-зависимости
startBot.py       # Точка входа с флагом голосового режима
```

## Запуск

Рабочее окружение контейнера — Python 3.11 с FFmpeg, Opus и зависимостями
из `requirements.txt`. Для локального запуска подготовьте такое же окружение;
команды выполняются из корня репозитория:

```powershell
python -m pip install -r requirements.txt
python startBot.py false
```

`start.bat` также запускает текстовый режим. Для голосового режима:

```powershell
python startBot.py true
```

Переменные окружения (также загружаются из `.env`): `DISCORD_TOKEN`,
`OPENAI_API_KEY` — ключ DeepSeek. Для голоса дополнительно нужны
`CLOUDFLARE_ACCOUNT_ID`, `CLOUDFLARE_API_TOKEN` и `GIGAAM_MODEL_DIR` — каталог
с `model.int8.onnx` и `tokens.txt`; по умолчанию `/data/models/gigaam-ctc`.
Голосовая сессия загружает локальную модель при подключении к каналу.

Модель `models/silero_vad.onnx` ищется относительно исходников независимо от cwd.
`.env`, `cookies.txt` для yt-dlp, временные WAV и загружаемая музыка используют
рабочий каталог. В Docker это `/data`; исходники находятся в `/app/dsbot`,
Silero — `/app/models`, GigaAM — `/opt/models/gigaam-ctc`.

Для Docker Compose задайте переменные окружения и `DSBOT_DATA_DIR` — абсолютный
путь к каталогу данных на хосте. `VOICE_ASSISTANT` выбирает `false` или `true`.
Команда контейнера остаётся `python /app/startBot.py <флаг>`.

## Проверки

В подготовленном окружении Python 3.11 с зависимостями приложения, FFmpeg и Opus:

```powershell
python -m unittest discover -s tests -v
```

Набор не требует токенов и сетевых запросов: внешние вызовы подменены;
проверки SDK, Opus, FFmpeg и Silero используют реальные библиотеки и модель.
Пропуски тестов из-за отсутствующих зависимостей не подтверждают работоспособность.
CI формирует coverage XML и передаёт его в SonarQube Cloud.

## Будущий MCP

При появлении конкретных требований модуль взаимодействия по MCP можно разместить
в `dsbot/mcp/`. Сейчас MCP не реализован и этот каталог не создаётся.
Chat, Music и управление ролями продолжают принимать объекты Discord;
общий API операций для MCP потребует отдельного проектирования.
