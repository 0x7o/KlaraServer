from fastapi import FastAPI, Request
from config import Config
from stt import STT
from tts import TTS

app = FastAPI()
s = STT(Config("config.json"))
t = TTS(Config("config.json"))


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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
