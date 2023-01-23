from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from config import Config


class Ner:
    def __init__(self, config: Config):
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.get_config("ner_model_name")
        )
        self.model = AutoModelForTokenClassification.from_pretrained(
            config.get_config("ner_model_name")
        ).to(config.get_config("ner_device"))
        self.nlp = pipeline(
            "ner",
            model=self.model,
            tokenizer=self.tokenizer,
            device=config.get_config("ner_device_index"),
        )

    def get_entities(self, text):
        entities = self.nlp(text)
        cleaned_entities = []
        for entity in entities:
            cleaned_entities.append({"entity": entity["entity"], "word": entity["word"]})
        return cleaned_entities