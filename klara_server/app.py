from fastapi import FastAPI, Request
from config import Config
from nllb import Translate
from stt import STT
from tts import TTS

app = FastAPI()
s = STT(Config("config.json"))
print("STT loaded")
t = TTS(Config("config.json"))
print("TTS loaded")
n = Translate(Config("config.json"))
print("NLLB loaded")


@app.get("/")
async def read_root():
    return {"Hello": "World"}


@app.post("/stt")
async def stt(request: Request):
    data = await request.json()
    return s.transcribe(data["audio"])


@app.post("/tts")
async def tts(request: Request):
    data = await request.json()
    return t.generate(data["text"])


@app.post("/nllb")
async def nllb(request: Request):
    data = await request.json()
    return n.transcribe(
        data["text"],
        data["src_lang"],
        data["tgt_lang"],
        data["gender_name"],
        data["gender_translation"],
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
