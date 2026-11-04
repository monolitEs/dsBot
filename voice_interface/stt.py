from vosk import KaldiRecognizer, Model
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
import json
import io
import asyncio
from .transcription import QuotaExceeded, SttError, Transcriber, run_blocking


class VoiceSession:
    """Models and quota state live only for one guild voice connection."""

    def __init__(self, cloud: Transcriber, local: Transcriber, detector_path: str):
        self.cloud, self.local = cloud, local
        self.detector_path = detector_path
        self.detector = self.listener = None
        self.closed = self.fallback = False
        self.lock, self.record_lock = asyncio.Lock(), asyncio.Lock()

    async def start(self):
        async with self.lock:
            if self.closed:
                raise SttError('Voice session is closed')
            if self.detector is None:
                try:
                    await self.cloud.start()
                    await self.local.start()
                    self.detector = await run_blocking(Model, self.detector_path)
                    print('STT models=loaded')
                except BaseException:
                    await self.local.close()
                    await self.cloud.close()
                    raise

    async def detect(self, wav):
        async with self.lock:
            if self.closed:
                return ''
            return await run_blocking(detect_words, wav, self.detector)

    async def transcribe(self, filename: str) -> str:
        async with self.lock:
            if self.closed:
                raise SttError('Voice session is closed')
            wav = await run_blocking(lambda: AudioSegment.from_wav(filename).set_frame_rate(16000)
                                     .export(io.BytesIO(), format='wav').getvalue())
            if not self.fallback:
                try:
                    return await self.cloud.transcribe(wav)
                except QuotaExceeded:
                    self.fallback = True
                    print('STT backend=local reason=quota')
            return await self.local.transcribe(wav)

    async def stop(self):
        if self.listener:
            listener, self.listener = self.listener, None
            listener.cancel()
            if listener.get_task():
                await asyncio.gather(listener.get_task(), return_exceptions=True)

    async def close(self):
        self.closed = True
        async with self.record_lock:
            await self.stop()
            async with self.lock:
                self.detector = None
                await self.local.close()
                await self.cloud.close()
                print('STT session=closed')




def detect_words(wf, model, framerate = 16000):
    """Just a function for taking an audio stream and retrieving words from VOSK kaldi"""
    
    audio = AudioSegment.from_file(wf, format="wav")
    audio = audio.set_frame_rate(16000)
    
    
    
    rec = KaldiRecognizer(model, framerate)
    rec.AcceptWaveform(audio.raw_data)
    
    result_dict = json.loads(rec.Result())
            
    
    return result_dict['text']

def speech_to_text(filename, model):
    
    audio = AudioSegment.from_wav(filename)
    
    nonsilent_ranges = detect_nonsilent(audio, min_silence_len=200, silence_thresh=-50)
    
    if nonsilent_ranges:
        first_sound_start = nonsilent_ranges[0][0]
        start_trim = max(0, first_sound_start - 500)

        trimmed_audio = audio[start_trim:]

        # Сохраняем в памяти (в формат WAV)
        trimmed_audio.export(filename, format="wav")
       
  
    result = model.transcribe(filename, language="ru")
    return result["text"]
    
    
    """
    model = Model(lang="en-gb")
    commands = str(commands).replace("'", '"')

    rec = KaldiRecognizer(model, framerate, 
    commands)

    transcription = []

    rec.SetWords(True)

    while True:
        data = wf.read(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            result_dict = json.loads(rec.Result())
            transcription.append(result_dict.get("text", ""))

    final_result = json.loads(rec.FinalResult())
    transcription.append(final_result.get("text", ""))

    transcription_text = ' '.join(transcription)
    return transcription_text
    """
