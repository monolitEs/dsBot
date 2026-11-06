# Pycord для голосового приёма

Проверенная пара: этот wheel Pycord и `davey==0.1.6` из requirements.txt.
Основа — PR3159, commit `10a5e8cf13bd8db21f95baf62cf866056d1e1091`.
`2.8.1.dev0+g10a5e8c` — локальное обозначение сборки, не официальный релиз.

175 Python-файлов SDK совпадают с проверенным тестовым образом r5.
Единственная правка upstream — список трёх подпакетов в pyproject.toml:
`discord.voice.receive`, `discord.voice.packets`, `discord.voice.utils`.
Точная правка в packaging-only.diff; crypto source не изменён.
Лицензия MIT в LICENSE; происхождение и хеши — в двух manifest JSON и SHA256SUMS.
Поле production_install=false в исходном provenance описывает проверку reference wheel
до переноса в проект, а не текущий способ установки.

## Установка

Из корня checkout: `python -m pip install -r requirements.txt`.
pip проверяет SHA256 wheel по фрагменту URL в requirements.txt.
Отдельная строка py-cord[voice] включает voice extras того же wheel;
такая запись проверена с pip 24 в контейнере и pip 25.3 на Windows.
Docker устанавливает из /app; рабочая папка запущенного бота остаётся /data.
Python 3.11 в CI и контейнере. Wheel Pycord pure Python; бинарный davey
pip выбирает для целевой платформы отдельно.

## Пересборка

Скачать архив по точному URL из source-manifest.json и проверить его SHA256.
Из корня проекта выполнить в окружении с Python 3.11:

```text
python docker/build_pycord_wheel.py PATH_TO_SOURCE_ARCHIVE --output OUTPUT_DIRECTORY
```

Скрипт проверяет хеш исходников, меняет только упаковку, собирает wheel и сверяет
полный набор и содержимое 175 Python-файлов с исходниками. Существующий wheel
с другими байтами не перезаписывается. При отличающемся хеше новой сборки сравнить
содержимое и build tools; не менять автоматически хеш в requirements.txt.
Сборка использует build isolation и диапазоны build tools из upstream pyproject.toml;
байтовая воспроизводимость между версиями инструментов не гарантирована.

## Ограничения

davey 0.1.6 выбран ради совместимости с этим Pycord. Исправления безопасности
и управления переходами ключей из 0.2 в эту комбинацию не входят.
Переход на другую версию требует проверки SDK/DAVE и живого голосового цикла.
Откат релиза — предыдущий образ; при необходимости запуск с voice=false
по существующему процессу владельца. Диагностика WAV/текстов r5 здесь отсутствует.
