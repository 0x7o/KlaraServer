from transformers import pipeline
from config import Config


class Intent:
    def __init__(self, config: Config):
        self.config = config
        self.pipe = pipeline(
            model=self.config.get_config("intent_model"),
            device=self.config.get_config("intent_device"),
        )

    def predict(self, text: str):
        result = self.pipe(
            text,
        )
        return result
