"""
REST API cho Qwen3-TTS — dùng khi muốn gọi từ app khác thay vì bấm trên web.

Chạy trên máy vast.ai:
    export API_KEY='mot-chuoi-bi-mat'
    uvicorn api_server:app --host 0.0.0.0 --port 8000

Trên Mac (đã có tunnel -L 8000:localhost:8000):
    curl -X POST http://localhost:8000/tts \
      -H 'Content-Type: application/json' \
      -H "X-API-Key: $API_KEY" \
      -d '{"mode":"custom_voice","text":"Hello there.","speaker":"Ryan","language":"English"}' \
      --output out.wav
"""

from __future__ import annotations

import io
import os
import threading

import soundfile as sf
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from engine import LANGUAGES, MODE_LABELS, SPEAKERS, OutOfVRAM, TTSEngine

API_KEY = os.getenv("API_KEY")

app = FastAPI(title="Qwen3-TTS", version="1.1")

# Khởi tạo lười: nếu tạo engine ngay lúc import, một máy thiếu GPU sẽ làm
# uvicorn chết trước khi kịp in ra lỗi gì có ích.
_engine: TTSEngine | None = None
_engine_lock = threading.Lock()


def get_engine() -> TTSEngine:
    global _engine
    with _engine_lock:
        if _engine is None:
            _engine = TTSEngine(
                device=os.getenv("TTS_DEVICE", "cuda:0"),
                use_06b=os.getenv("USE_06B", "").lower() in {"1", "true", "yes"},
            )
        return _engine


def check_key(x_api_key: str | None = Header(default=None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(401, "Sai hoặc thiếu header X-API-Key.")


class TTSRequest(BaseModel):
    mode: str = Field("custom_voice", description="custom_voice | voice_design | voice_clone")
    text: str
    language: str = "Auto"
    speaker: str | None = "Ryan"
    instruct: str | None = None
    ref_audio: str | None = Field(None, description="Đường dẫn file, URL, hoặc chuỗi base64")
    ref_text: str | None = None
    x_vector_only: bool = False
    max_chars: int = 220
    max_new_tokens: int = 2048
    temperature: float | None = None
    top_p: float | None = None
    seed: int | None = None


@app.get("/healthz")
def healthz():
    try:
        eng = get_engine()
        return {"ok": True, "model": eng.model_id, "device": eng.vram_report()}
    except RuntimeError as exc:
        return {"ok": False, "error": str(exc)}


@app.get("/voices")
def voices():
    return {"modes": MODE_LABELS, "speakers": SPEAKERS, "languages": LANGUAGES}


@app.post("/unload", dependencies=[Depends(check_key)])
def unload():
    return {"message": get_engine().unload()}


@app.post("/tts", dependencies=[Depends(check_key)])
def tts(req: TTSRequest):
    try:
        sr, audio = get_engine().synthesize(**req.model_dump())
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except OutOfVRAM as exc:
        raise HTTPException(507, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(500, str(exc)) from exc

    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    return Response(
        content=buf.getvalue(),
        media_type="audio/wav",
        headers={"Content-Disposition": 'attachment; filename="speech.wav"'},
    )
