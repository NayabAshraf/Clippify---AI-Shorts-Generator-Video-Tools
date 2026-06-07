"""
services/transcriber_service.py
Microservice: Transcriber (using OpenAI Whisper)
Port: 8002
"""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
import whisper
import shutil
import os
import logging
import uuid
from pathlib import Path
import tempfile
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Import from shared config
from shared.config import (
    TRANSCRIBER_PORT,
    INTERNAL_TOKEN,
    SHARED_STORAGE_ROOT,
    get_ffmpeg_path,
    validate_ffmpeg
)
from shared.http_client import require_internal_token

# ── FFMPEG Setup using config ───────────────────────────────────
# Set FFmpeg path from config
FFMPEG_PATH = get_ffmpeg_path()
os.environ["PATH"] = os.path.dirname(FFMPEG_PATH) + os.pathsep + os.environ.get("PATH", "")

# Validate FFmpeg is working
validate_ffmpeg()

# ── Configuration ───────────────────────────────────
MODEL_SIZE = "base"
ALLOWED_EXTENSIONS = {".mp3", ".wav", ".m4a", ".mp4"}

# Force CPU to avoid CUDA issues
DEVICE = "cpu"
USE_FP16 = False

# ── Logging Setup ───────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ── App Initialization ──────────────────────────────
app = FastAPI(title="Transcriber Service", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load Model (only once) ──────────────────────────
logging.info(f"Loading Whisper model: {MODEL_SIZE} on {DEVICE}")
model = whisper.load_model(MODEL_SIZE, device=DEVICE)
logging.info("Model loaded successfully")

# ── Helper Function ─────────────────────────────────
def format_segments(segments):
    return [
        {
            "start": round(seg["start"], 2),
            "end": round(seg["end"], 2),
            "text": seg["text"].strip()
        }
        for seg in segments
    ]

# ── API Endpoint ────────────────────────────────────
@app.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    include_timestamps: bool = Form(True),
    _ = Depends(require_internal_token)
):
    temp_file_path = None
    
    try:
        logging.info(f"Received file: {file.filename}")
        
        # Validate file extension
        ext = Path(file.filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type '{ext}'. Allowed: {ALLOWED_EXTENSIONS}"
            )
        
        # Create temp directory in shared storage or system temp
        temp_dir = Path(SHARED_STORAGE_ROOT) / "temp_transcribe"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Create unique filename
        temp_file_path = temp_dir / f"{uuid.uuid4().hex}{ext}"
        
        # Save file
        with open(temp_file_path, 'wb') as f:
            content = await file.read()
            f.write(content)
        
        logging.info(f"Saved to temp file: {temp_file_path}")
        
        # Verify file exists and has content
        if not temp_file_path.exists():
            raise HTTPException(500, "Failed to create temporary file")
        
        file_size = temp_file_path.stat().st_size
        logging.info(f"File size: {file_size} bytes")
        
        if file_size == 0:
            raise HTTPException(400, "Uploaded file is empty")
        
        # Transcribe
        logging.info("Starting transcription...")
        result = model.transcribe(
            str(temp_file_path),
            fp16=USE_FP16,
            word_timestamps=include_timestamps
        )
        
        logging.info("Transcription completed successfully")
        
        # Format response
        response_data = {
            "transcript": result["text"].strip(),
        }
        
        if include_timestamps:
            response_data["segments"] = format_segments(result["segments"])
        
        return JSONResponse(response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Transcription error: {str(e)}", exc_info=True)
        raise HTTPException(500, f"Transcription failed: {str(e)}")
    finally:
        # Clean up temp file
        if temp_file_path and temp_file_path.exists():
            try:
                temp_file_path.unlink()
                logging.info(f"Cleaned up temp file: {temp_file_path}")
            except Exception as e:
                logging.warning(f"Failed to delete temp file: {e}")

@app.get("/health")
async def health():
    return {"status": "ok", "service": "transcriber", "ffmpeg_configured": True}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=TRANSCRIBER_PORT)