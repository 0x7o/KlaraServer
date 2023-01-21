from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from config import Config


class Translate:
    def __init__(self, config: Config):
        self.config = config
        self.model, self.tokenizer = self.load_model()

    def load_model(self):
        model = AutoModelForSeq2SeqLM.from_pretrained(
            self.config.get_config("nllb_model")
        ).to(self.config.get_config("nllb_device"))
        tokenizer = AutoTokenizer.from_pretrained(self.config.get_config("nllb_model"))
        return model, tokenizer

    def transcribe(self, text, src_lang, tgt_lang, gender_name, gender_translation):
        translator = pipeline(
            "translation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.config.get_config("nllb_device_index"),
            src_lang=src_lang,
            tgt_lang=tgt_lang,
        )
        text = translator(f"{gender_name}: {text}", max_length=400)
        text.replace(gender_translation, "")
        return text
