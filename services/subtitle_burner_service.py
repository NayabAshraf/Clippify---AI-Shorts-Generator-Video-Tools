"""
services/subtitle_burner_service.py
=====================================
Microservice: Subtitle Burner
Port: 8005  (SUBTITLE_BURNER_PORT)

Endpoints
---------
POST /burn-subtitles
    Body: JSON {"clip_paths": ["<path1>", "<path2>", ...]}
    Transcribes each clip with faster-whisper (word-level),
    generates an ASS karaoke subtitle file, and burns it in
    with FFmpeg.
    Returns paths to the subtitled clips on shared storage.

POST /burn-video-direct
    Upload video file directly, transcribe and burn subtitles in one step.
    Returns path to the subtitled video.

GET /health
    Returns {"status": "ok"}.

Run
---
    uvicorn services.subtitle_burner_service:app --port 8005 --reload
"""

import shutil
import subprocess
import tempfile
import time
import uuid
import warnings
from pathlib import Path
from typing import Optional

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import Depends, FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from shared.config import FFMPEG_BIN, SHARED_STORAGE_ROOT
from shared.http_client import require_internal_token

warnings.filterwarnings("ignore", category=UserWarning)

app = FastAPI(
    title="Subtitle Burner Service",
    description="Burns dynamic karaoke-style captions into video clips.",
    version="2.0.0",
)

SUBTITLED_OUT_DIR = Path(SHARED_STORAGE_ROOT) / "subtitled"
SUBTITLED_OUT_DIR.mkdir(parents=True, exist_ok=True)

TEMP_UPLOAD_DIR = Path(SHARED_STORAGE_ROOT) / "temp_subtitle_uploads"
TEMP_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ── Styling constants ─────────────────────────────────────────────────────────
FONT_NAME      = "Arial"
FONT_SIZE      = 52
WORDS_PER_LINE = 4
COL_ACTIVE     = "&H0000FFFF"   # Yellow
COL_SPOKEN     = "&H00CCCCCC"   # Light gray
COL_PENDING    = "&H00FFFFFF"   # White
COL_OUTLINE    = "&H00000000"
COL_SHADOW     = "&HAA000000"
MARGIN_V       = 130

_whisper_model = None


def _get_model():
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel
        _whisper_model = WhisperModel("small", device="cpu", compute_type="int8")
    return _whisper_model


# ── ASS helpers ───────────────────────────────────────────────────────────────

def _ts(seconds: float) -> str:
    cs   = int(round((seconds % 1) * 100))
    secs = int(seconds)
    m, s = divmod(secs, 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _ass_header() -> str:
    return (
        "[Script Info]\nScriptType: v4.00+\nPlayResX: 1080\nPlayResY: 1920\nWrapStyle: 0\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Default,{FONT_NAME},{FONT_SIZE},{COL_PENDING},&H0000FFFF,"
        f"{COL_OUTLINE},{COL_SHADOW},1,0,1,3,2,2,40,40,{MARGIN_V},1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )


def _colorize(text: str, color: str) -> str:
    return f"{{\\c{color}}}{text}{{\\c{COL_PENDING}}}"


def _build_ass(words: list[dict]) -> str:
    groups = [words[i: i + WORDS_PER_LINE] for i in range(0, len(words), WORDS_PER_LINE)]
    events = []
    for group in groups:
        for active_idx, active_word in enumerate(group):
            parts = []
            for i, w in enumerate(group):
                if i < active_idx:
                    parts.append(_colorize(w["word"], COL_SPOKEN))
                elif i == active_idx:
                    parts.append(_colorize(w["word"], COL_ACTIVE))
                else:
                    parts.append(w["word"])
            events.append(
                f"Dialogue: 0,{_ts(active_word['start'])},{_ts(active_word['end'])},"
                f"Default,,0,0,0,,{' '.join(parts)}\n"
            )
    return _ass_header() + "".join(events)


def _burn_one(clip_path: Path, out_path: Path, model=None) -> tuple[bool, Optional[list[dict]]]:
    """Transcribe clip, build ASS, burn subtitles with FFmpeg.
    Returns (success, word_timestamps)"""
    if model is None:
        model = _get_model()
    
    segs, _ = model.transcribe(str(clip_path), word_timestamps=True)

    words = []
    for seg in segs:
        if seg.words:
            for w in seg.words:
                if w.word.strip():
                    words.append(
                        {"word": w.word.strip(), "start": float(w.start), "end": float(w.end)}
                    )

    if not words:
        return False, None

    # Build and save ASS file
    ass_path = out_path.with_suffix(".ass")
    ass_content = _build_ass(words)
    ass_path.write_text(ass_content, encoding="utf-8")

    # Burn subtitles into video
    ass_escaped = str(ass_path).replace("\\", "/").replace(":", "\\:")
    cmd = [
        FFMPEG_BIN, "-y",
        "-i", str(clip_path),
        "-vf", f"subtitles='{ass_escaped}'",
        "-c:v", "libx264", "-preset", "superfast", "-crf", "23",
        "-c:a", "copy",
        str(out_path),
    ]

    result = subprocess.run(cmd, capture_output=True)
    ass_path.unlink(missing_ok=True)
    
    if result.returncode == 0:
        return True, words
    return False, None


def _extract_audio_from_video(video_path: Path) -> Path:
    """Extract audio from video for preview."""
    audio_path = video_path.with_suffix(".mp3")
    cmd = [
        FFMPEG_BIN, "-y",
        "-i", str(video_path),
        "-vn",
        "-acodec", "libmp3lame",
        "-ab", "128k",
        "-ar", "16000",
        "-ac", "1",
        str(audio_path)
    ]
    subprocess.run(cmd, capture_output=True)
    return audio_path if audio_path.exists() else None


# ── Request schemas ────────────────────────────────────────────────────────────

class BurnRequest(BaseModel):
    clip_paths: list[str]
    """List of absolute paths to raw clips on shared storage."""
    job_id: str | None = None


class BurnResponse(BaseModel):
    job_id: str
    subtitled: list[dict]
    count: int
    duration_s: float


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.post(
    "/burn-subtitles",
    summary="Burn karaoke subtitles into video clips",
    dependencies=[Depends(require_internal_token)],
)
async def burn_subtitles(req: BurnRequest) -> JSONResponse:
    if not req.clip_paths:
        raise HTTPException(status_code=422, detail="clip_paths must not be empty.")

    job_id  = req.job_id or uuid.uuid4().hex[:8]
    job_dir = SUBTITLED_OUT_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    created: list[dict] = []
    t0 = time.time()
    model = _get_model()

    for clip_str in req.clip_paths:
        clip = Path(clip_str)
        if not clip.exists():
            continue

        out_path = job_dir / f"{clip.stem}_subtitled.mp4"
        ok, words = _burn_one(clip, out_path, model)

        if ok:
            created.append(
                {
                    "raw_clip": str(clip),
                    "subtitled_clip": str(out_path),
                    "word_count": len(words) if words else 0
                }
            )

    return JSONResponse(
        status_code=200,
        content={
            "job_id":     job_id,
            "subtitled":  created,
            "count":      len(created),
            "duration_s": round(time.time() - t0, 2),
        },
    )


@app.post("/burn-video-direct")
async def burn_video_direct(
    file: UploadFile = File(...),
    job_id: Optional[str] = Form(None),
) -> JSONResponse:
    """
    Upload a video directly and burn subtitles into it.
    This endpoint:
    1. Saves the uploaded video
    2. Transcribes it using faster-whisper
    3. Generates karaoke-style ASS subtitles
    4. Burns subtitles into the video
    5. Returns the path to the subtitled video
    """
    temp_files = []
    
    try:
        # ── Stage 1: Save uploaded video ──────────────────────────────────────
        suffix = Path(file.filename or "video.mp4").suffix.lower()
        if suffix not in ['.mp4', '.mov', '.avi', '.mkv', '.webm']:
            raise HTTPException(400, f"Unsupported video format: {suffix}")
        
        video_id = uuid.uuid4().hex[:8]
        temp_video = TEMP_UPLOAD_DIR / f"temp_video_{video_id}{suffix}"
        
        with open(temp_video, "wb") as fh:
            shutil.copyfileobj(file.file, fh)
        temp_files.append(temp_video)
        
        # ── Stage 2: Transcribe and burn subtitles ────────────────────────────
        t0 = time.time()
        model = _get_model()
        
        # Generate output path
        job_dir = SUBTITLED_OUT_DIR / (job_id or video_id)
        job_dir.mkdir(parents=True, exist_ok=True)
        output_path = job_dir / f"{temp_video.stem}_subtitled{suffix}"
        
        # Burn subtitles
        success, words = _burn_one(temp_video, output_path, model)
        
        if not success:
            raise HTTPException(500, "Failed to burn subtitles into video")
        
        # ── Stage 3: Get transcription text for preview ───────────────────────
        transcription_text = " ".join([w["word"] for w in words]) if words else ""
        
        # Extract audio for preview (optional)
        audio_path = None
        try:
            audio_path = _extract_audio_from_video(temp_video)
            if audio_path:
                temp_files.append(audio_path)
        except Exception:
            pass
        
        elapsed = time.time() - t0
        
        # ── Stage 4: Return results ───────────────────────────────────────────
        return JSONResponse({
            "status": "success",
            "job_id": job_id or video_id,
            "original_filename": file.filename,
            "input_video": str(temp_video),
            "output_video": str(output_path),
            "subtitled_video": str(output_path),
            "word_count": len(words) if words else 0,
            "transcription": transcription_text[:500] + "..." if len(transcription_text) > 500 else transcription_text,
            "duration_s": round(elapsed, 2),
            "word_timestamps": words[:50] if words else []  # Return first 50 words for preview
        })
        
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, f"Subtitle burning failed: {str(exc)}")
    finally:
        # Clean up temporary files (keep the original if needed, but clean on next run)
        for temp_file in temp_files:
            try:
                if temp_file.exists():
                    # Don't delete immediately, schedule for later cleanup
                    pass
            except Exception:
                pass


@app.post("/burn-with-custom-style")
async def burn_with_custom_style(
    file: UploadFile = File(...),
    font_size: int = Form(52),
    font_color: str = Form("yellow"),
    words_per_line: int = Form(4),
    job_id: Optional[str] = Form(None),
) -> JSONResponse:
    """
    Burn subtitles with custom styling options.
    """
    temp_files = []
    
    try:
        # Override global constants for this request
        global FONT_SIZE, WORDS_PER_LINE, COL_ACTIVE
        
        original_font_size = FONT_SIZE
        original_words_per_line = WORDS_PER_LINE
        original_col_active = COL_ACTIVE
        
        try:
            FONT_SIZE = font_size
            WORDS_PER_LINE = words_per_line
            
            # Map color names to ASS color codes
            color_map = {
                "yellow": "&H0000FFFF",
                "white": "&H00FFFFFF",
                "red": "&H000000FF",
                "blue": "&H00FF0000",
                "green": "&H0000FF00",
                "cyan": "&H00FFFF00",
                "magenta": "&H00FF00FF",
            }
            COL_ACTIVE = color_map.get(font_color.lower(), "&H0000FFFF")
            
            # Process video
            suffix = Path(file.filename or "video.mp4").suffix.lower()
            video_id = uuid.uuid4().hex[:8]
            temp_video = TEMP_UPLOAD_DIR / f"temp_video_{video_id}{suffix}"
            
            with open(temp_video, "wb") as fh:
                shutil.copyfileobj(file.file, fh)
            temp_files.append(temp_video)
            
            # Burn subtitles
            job_dir = SUBTITLED_OUT_DIR / (job_id or video_id)
            job_dir.mkdir(parents=True, exist_ok=True)
            output_path = job_dir / f"{temp_video.stem}_subtitled{suffix}"
            
            model = _get_model()
            success, words = _burn_one(temp_video, output_path, model)
            
            if not success:
                raise HTTPException(500, "Failed to burn subtitles")
            
            return JSONResponse({
                "status": "success",
                "job_id": job_id or video_id,
                "output_video": str(output_path),
                "subtitled_video": str(output_path),
                "settings_used": {
                    "font_size": font_size,
                    "font_color": font_color,
                    "words_per_line": words_per_line
                },
                "duration_s": 0
            })
            
        finally:
            # Restore original constants
            FONT_SIZE = original_font_size
            WORDS_PER_LINE = original_words_per_line
            COL_ACTIVE = original_col_active
            
    except Exception as exc:
        raise HTTPException(500, f"Failed: {str(exc)}")
    finally:
        for temp_file in temp_files:
            try:
                if temp_file.exists():
                    pass
            except Exception:
                pass


@app.get("/health", summary="Health check")
async def health() -> dict:
    return {"status": "ok", "service": "subtitle_burner"}