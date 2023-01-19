import torch
import base64
import soundfile as sf
from config import Config
from IPython.display import Audio


class TTS:
    def __init__(self, config: Config):
        self.config = config
        model, text = torch.hub.load(
            repo_or_dir="snakers4/silero-models",
            model="silero_tts",
            language=self.config.get_config("tts_language"),
            speaker=self.config.get_config("tts_model"),
        )
        model.to(self.config.get_config("tts_device"))
        self.model = model

    def generate(self, text: str):
        audio = self.model.apply_tts(
            text=text,
            speaker=self.config.get_config("tts_speaker"),
            sample_rate=self.config.get_config("tts_sample_rate"),
            put_accent=True,
            put_yo=True,
        )
        audio = Audio(audio, rate=self.config.get_config("tts_sample_rate"))
        base64_audio = base64.b64encode(audio.data)
        return {"audio": base64_audio}


if __name__ == "__main__":
    config = Config("config.json")
    tts = TTS(config)
    base64_audio = tts.generate("Привет, я Клара")
    print(base64_audio)
