import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass(frozen=True)
class ConversationEvent:
    event_id: str
    session_id: str | None
    guild_id: int | None
    channel_id: int
    source_channel_id: int
    author_id: int
    author_name: str
    source: str
    captured_at: datetime
    sequence: int


@dataclass
class HistoryEntry:
    event: ConversationEvent
    settings_version: int
    content: str | None = None
    reply: str | None = None
    settled: asyncio.Event = field(default_factory=asyncio.Event)
    preceding: "list[HistoryEntry] | None" = field(default=None, repr=False, compare=False)


class ConversationHistory:
    MAX_EVENTS = 256

    def __init__(self, key):
        self.key = key
        self.entries = {}
        self.sequence = 0
        self.cleared_through = 0

    def reserve(self, message, session_id, settings_version):
        sequence = self.sequence + 1
        event_id = f'voice:{session_id}:{sequence}' if session_id else f'text:{message.id}'
        if event_id in self.entries:
            return None
        if len(self.entries) == self.MAX_EVENTS:
            oldest = next((key for key, entry in self.entries.items() if entry.settled.is_set()), None)
            if oldest is None:
                print(f'Chat: history_full channel={self.key[1]} event={event_id}')
                return None
            del self.entries[oldest]

        self.sequence = sequence
        event = ConversationEvent(
            event_id, session_id, *self.key, message.channel.id, message.author.id,
            message.author.name, 'voice' if session_id else 'text', datetime.now(timezone.utc), sequence)
        entry = HistoryEntry(event, settings_version)
        if session_id:
            entry.preceding = list(self.entries.values())
        self.entries[event_id] = entry
        return entry

    async def snapshot(self, request):
        # Keep the selected entries even if later messages evict them while we wait.
        preceding = request.preceding
        request.preceding = None
        if preceding is None:
            preceding = [entry for entry in self.entries.values()
                         if entry.event.sequence < request.event.sequence]
        for entry in preceding:
            await entry.settled.wait()

        messages = []
        for entry in [*preceding, request]:
            if entry.content is not None:
                author = entry.event
                messages.append({'role': 'user', 'content':
                                 f'[id={author.author_id}, name={author.author_name}] {entry.content}'})
            if entry.reply is not None:
                messages.append({'role': 'assistant', 'content': entry.reply})
        return messages

    def clear(self):
        self.cleared_through = self.sequence
        for entry in self.entries.values():
            entry.preceding = None
            entry.settled.set()
        self.entries.clear()
