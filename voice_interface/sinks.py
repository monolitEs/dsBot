from discord import sinks
import io
import threading

class StreamSink(sinks.WaveSink):
    # Pycord 2.8's reader expects the sink event/tree protocol even for a leaf.
    __sink_listeners__ = ()

    def __init__(self, *, filters=None):
        super().__init__(filters=filters)
        self._buffer_lock = threading.Lock()
        self._listening = True
        self.received_frames = 0
        self.unassigned_frames = 0

    def is_opus(self):
        return False  # Ask PacketDecoder for PCM, preserving the existing STT input.

    def walk_children(self):
        return iter(())

    def write(self, data, user):
        self.received_frames += 1
        if user is not None:
            self._write_pcm(data.pcm, user.id)
        else:
            self.unassigned_frames += 1

    @sinks.Filters.container
    def _write_pcm(self, data, user):
        with self._buffer_lock:
            if self.finished or not self._listening:
                return
            if user not in self.audio_data:
                self.audio_data[user] = sinks.AudioData(io.BytesIO())
            self.audio_data[user].write(data)

    def set_listening(self, enabled: bool):
        """Discard queued PCM at both boundaries of a playback interval."""
        with self._buffer_lock:
            self._listening = enabled
            for audio in self.audio_data.values():
                audio.file.seek(0)
                audio.file.truncate(0)

    def read(self, user_id) -> bytes:
        """Consume this participant's pending PCM without retaining its history."""
        with self._buffer_lock:
            audio = self.audio_data.get(user_id)
            if audio is None:
                raise KeyError("User not found")
            data = audio.file.getvalue()
            audio.file.seek(0)
            audio.file.truncate(0)
            return data

    def cleanup(self):
        # Streaming consumers own the drained PCM; no final WAV is needed.
        with self._buffer_lock:
            self.finished = True
            for audio in self.audio_data.values():
                audio.file.close()
            self.audio_data.clear()
