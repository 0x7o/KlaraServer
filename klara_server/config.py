import os
import json


class Config:
    def __init__(self, config_file):
        self.config_file = config_file
        self.config = self.load_config()

    def load_config(self):
        if os.path.exists(self.config_file):
            with open(self.config_file, "r") as f:
                return json.load(f)
        else:
            self.create_config()
            return self.load_config()

    def get_config(self, key):
        try:
            return self.config[key]
        except KeyError:
            assert False, f"Key {key} not found in config"

    def edit_config(self, key, value):
        self.config[key] = value
        with open(self.config_file, "w") as f:
            json.dump(self.config, f, indent=4)

    def create_config(self):
        default_config = {
            "whisper_base_model": "small",
            "whisper_device": "cuda:0",
            "whisper_model_path": "small-ru.pt",
            "whisper_language": "ru",
            "fp16": False,
            "sample_rate": 16000,
            "tts_model": "v3_1_ru",
            "tts_language": "ru",
            "tts_device": "cuda:1",
            "tts_sample_rate": 48000,
            "tts_speaker": "kseniya",
            "intent_model": "0x7194633/rubert-base-massive",
            "intent_device": "cuda:0",
            "ner_model_name": "cartesinus/xlm-r-base-amazon-massive-slot",
            "ner_device": "cuda:0",
        }
        with open(self.config_file, "w") as f:
            json.dump(default_config, f, indent=4)


if __name__ == "__main__":
    config = Config("config.json")
    print(config.get_config("model_path"))
