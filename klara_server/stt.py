import base64
import numpy as np
import soundfile as sf
from config import Config
from transformers import pipeline


class STT:
    def __init__(self, config: Config):
        self.config = config
        self.stt = pipeline(
            "automatic-speech-recognition",
            model=self.config.get_config("stt_model"),
            device=self.config.get_config("stt_device"),
        )

    def base64_to_wav(self, base64_string):
        wav_bytes = base64.b64decode(base64_string)
        wav = np.frombuffer(wav_bytes, dtype=np.int16)
        sf.write("temp.wav", wav, self.config.get_config("sample_rate"))

    def transcribe(self, base64_string):
        self.base64_to_wav(base64_string)
        out = self.stt(
            "temp.wav",
        )

        return out["text"]
