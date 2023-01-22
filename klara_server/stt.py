import torch
import base64
import numpy as np
from config import Config
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

    def transcribe(self, base64_string):
        wav_bytes = base64.b64decode(base64_string)
        input_speech = np.frombuffer(wav_bytes, dtype=np.float32)
        input_speech = torch.from_numpy(input_speech)
        self.model.config.forced_decoder_ids = self.processor.get_decoder_prompt_ids(
            language=self.config.get_config("whisper_language"), task="transcribe"
        )
        input_features = self.processor(
            input_speech,
            return_tensors="pt",
            sampling_rate=self.config.get_config("sample_rate"),
        ).input_features.to(self.config.get_config("whisper_device"))
        predicted_ids = self.model.generate(
            input_features,
            return_dict_in_generate=True,
        )
        transcription = self.processor.batch_decode(predicted_ids)
        return transcription[0]
