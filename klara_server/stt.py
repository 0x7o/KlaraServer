import torch
import base64
import numpy as np
import soundfile as sf
from config import Config
from scipy.io.wavfile import read
from transformers import WhisperProcessor, WhisperForConditionalGeneration


class STT:
    def __init__(self, config: Config):
        self.config = config
        self.model, self.processor = self.load_model()

    def load_model(self):
        processor = WhisperProcessor.from_pretrained(
            self.config.get_config("whisper_model")
        )
        model = WhisperForConditionalGeneration.from_pretrained(
            self.config.get_config("whisper_model")
        ).to(self.config.get_config("whisper_device"))

        return model, processor

    def base64_to_wav(self, base64_string):
        wav_bytes = base64.b64decode(base64_string)
        wav = np.frombuffer(wav_bytes, dtype=np.int16)
        sf.write("temp.wav", wav, self.config.get_config("sample_rate"))

    def transcribe(self, base64_string):
        self.base64_to_wav(base64_string)
        rate, data = read("temp.wav")
        input_speech = np.array(data, dtype=np.float32)
        input_speech = torch.from_numpy(input_speech)
        self.model.config.forced_decoder_ids = self.processor.get_decoder_prompt_ids(
            language=self.config.get_config("whisper_language"), task="transcribe"
        )
        input_features = self.processor(
            input_speech,
            return_tensors="pt",
            sampling_rate=self.config.get_config("sample_rate"),
        ).input_features.to(self.config.get_config("whisper_device"))
        predicted_ids = self.model.generate(input_features, temperature=1.0, top_p=0.9)
        transcription = self.processor.batch_decode(predicted_ids)
        return transcription[0]
