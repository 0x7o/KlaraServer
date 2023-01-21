from config import Config
import requests


class Translate:
    def __init__(self, config: Config):
        self.config = config

    def transcribe(self, text, src_lang, tgt_lang, gender_name, gender_translation):
        url = "https://cloud.yandex.ru/api/translate/translate"
        payload = {
            "sourceLanguageCode": src_lang,
            "targetLanguageCode": tgt_lang,
            "texts": [f"{gender_name}: {text}"],
        }
        headers = {"Content-Type": "application/json"}
        response = requests.request("POST", url, json=payload, headers=headers)
        translation = response.json()["translations"][0]["text"]
        translation = translation.replace(gender_translation, "")
        return translation
