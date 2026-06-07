"""
services/audio_extractor_service.py
=====================================
Microservice: Audio Extractor
Port: 8001  (AUDIO_EXTRACTOR_PORT)

Endpoints
---------
POST /extract-audio
    Accepts a video file upload (multipart/form-data).
    Runs FFmpeg to extract mono 16kHz MP3 audio.
    Returns JSON with the path to the extracted audio file on shared storage.

GET /health
    Returns {"status": "ok"}.

Run
---
    uvicorn services.audio_extractor_service:app --port 8001 --reload
"""

import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shared.config import FFMPEG_BIN, SHARED_STORAGE_ROOT
from shared.http_client import require_internal_token

app = FastAPI(
    title="Audio Extractor Service",
    description="Extracts mono 16kHz MP3 audio from a video file.",
    version="1.0.0",
)

AUDIO_OUT_DIR = Path(SHARED_STORAGE_ROOT) / "audio"
AUDIO_OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Endpoint ──────────────────────────────────────────────────────────────────

@app.post(
    "/extract-audio",
    summary="Extract audio from a video file",
    dependencies=[Depends(require_internal_token)],
)
async def extract_audio(
    file: UploadFile = File(..., description="Source video file"),
) -> JSONResponse:
    """
    Saves the uploaded video to a temp path, runs FFmpeg to strip audio,
    writes the MP3 to shared storage, and returns its path + metadata.
    """
    job_id = uuid.uuid4().hex[:8]
    suffix = Path(file.filename or "video.mp4").suffix.lower() or ".mp4"

    # Save upload to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        video_path = Path(tmp.name)

    try:
        audio_path = AUDIO_OUT_DIR / f"{job_id}_audio.mp3"
        t0 = time.time()

        cmd = [
            FFMPEG_BIN,
            "-y",
            "-i", str(video_path),
            "-ac", "1",       # mono
            "-ar", "16000",   # 16 kHz — Whisper's preferred rate
            str(audio_path),
        ]

        result = subprocess.run(cmd, capture_output=True)

        if result.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail=f"FFmpeg error: {result.stderr.decode(errors='replace')}",
            )

        elapsed = time.time() - t0

        return JSONResponse(
            status_code=200,
            content={
                "job_id":     job_id,
                "audio_path": str(audio_path),
                "duration_s": round(elapsed, 2),
            },
        )

    finally:
        video_path.unlink(missing_ok=True)


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health", summary="Health check")
async def health() -> dict:
    return {"status": "ok", "service": "audio_extractor"}