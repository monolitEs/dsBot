from discord import sinks
import io

class StreamAudioData(sinks.AudioData):
    def __init__(self, file):
        super().__init__(file)
    
        self.last_read_byte = 0

class StreamSink(sinks.WaveSink):
    # Pycord 2.8's reader expects the sink event/tree protocol even for a leaf.
    __sink_listeners__ = ()

    def __init__(self, *, filters=None):
        super().__init__(filters=filters)
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
        if self.received_frames == 1 or self.received_frames % 250 == 0:
            print(f'STT receive frames={self.received_frames} '
                  f'unassigned={self.unassigned_frames} pcm_bytes={len(data.pcm)}')

    @sinks.Filters.container
    def _write_pcm(self, data, user):
        if user not in self.audio_data:
            file = io.BytesIO()
            self.audio_data.update({user: StreamAudioData(file)})

        file = self.audio_data[user]
        file.write(data)

    def read(self, user_id, full=False, starting_point=0) -> bytearray:
        audio = self.audio_data.get(user_id)
        if not audio:
            raise KeyError("User not found")
        
        data = audio.file.getvalue()

        if not full:
            if not starting_point:
                starting_point = audio.last_read_byte
            data = data[starting_point:]
        
        audio.last_read_byte = len(audio.file.getvalue())

        return data
