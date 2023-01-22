import base64
import whisper
import numpy as np
import soundfile as sf
from config import Config


class STT:
    def __init__(self, config: Config):
        self.config = config
        self.model, self.options = self.load_model()

    def load_model(self):
        model = whisper.load_model(
            self.config.get_config("whisper_model"),
            device=self.config.get_config("whisper_device"),
        )
        options = whisper.DecodingOptions(
            language=self.config.get_config("whisper_language"),
            fp16=self.config.get_config("fp16"),
            task="transcribe",
        )
        return model, options

    def base64_to_wav(self, base64_string):
        wav_bytes = base64.b64decode(base64_string)
        wav = np.frombuffer(wav_bytes, dtype=np.int16)
        sf.write("temp.wav", wav, self.config.get_config("sample_rate"))

    def transcribe(self, base64_string):
        self.base64_to_wav(base64_string)
        audio = whisper.load_audio("temp.wav")
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(self.model.device)
        transcript = whisper.decode(self.model, mel, self.options)
        return transcript.text
