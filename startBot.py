import os
import sys
from dsbot.bot import init

def main(voice_assistant):
    voice_assistant_bool = False
    if voice_assistant.lower() == "true":
        voice_assistant_bool = True

    bot = init(voice_assistant_bool)
    bot.run(os.getenv("DISCORD_TOKEN"))

if __name__ == "__main__":
    if len(sys.argv) < 1:
        print("Wrong arguments count")
        sys.exit(1)

    main(sys.argv[1])

