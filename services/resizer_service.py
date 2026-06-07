"""
resizer_service.py — Video Resizer Microservice
================================================
Converts landscape videos to portrait (9:16) format
WITHOUT CROPPING (uses padding / letterboxing).

Run:
    uvicorn resizer_service:app --host 0.0.0.0 --port 8006
"""

import json
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

# ── Config ────────────────────────────────────────────────────────────────────

INTERNAL_TOKEN = os.getenv("INTERNAL_TOKEN", "dev-secret-change-me")
SHARED_OUTPUT = Path(os.getenv("SHARED_OUTPUT", "/tmp/reels_shared/resized"))
SHARED_OUTPUT.mkdir(parents=True, exist_ok=True)

PORTRAIT_PRESETS = {
    "tiktok": {"width": 1080, "height": 1920, "label": "TikTok / Reels (9:16)"},
    "instagram": {"width": 1080, "height": 1350, "label": "Instagram Feed (4:5)"},
    "square": {"width": 1080, "height": 1080, "label": "Square (1:1)"},
}

app = FastAPI(
    title="Video Resizer Service",
    description="Converts landscape videos to portrait format using padding (no cropping).",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Auth ──────────────────────────────────────────────────────────────────────

def require_token(x_internal_token: str | None) -> None:
    if x_internal_token != INTERNAL_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")


# ── Helpers ───────────────────────────────────────────────────────────────────

def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def get_dimensions(video_path: str) -> tuple[int, int]:
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr[:300]}")

    streams = json.loads(result.stdout).get("streams", [])
    video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)

    if not video_stream:
        raise RuntimeError("No video stream found")

    return int(video_stream["width"]), int(video_stream["height"])


def get_orientation(w: int, h: int) -> str:
    if w > h:
        return "landscape"
    elif h > w:
        return "portrait"
    return "square"


def _hw_accel_args() -> list[str]:
    """
    Use NVENC if реально available, otherwise fallback to CPU.
    """

    try:
        test = subprocess.run(
            [
                "ffmpeg",
                "-f", "lavfi",
                "-i", "nullsrc=s=128x128:d=1",
                "-c:v", "h264_nvenc",
                "-f", "null",
                "-"
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5
        )
        if test.returncode == 0:
            return ["-c:v", "h264_nvenc", "-preset", "p1", "-tune", "ll"]
    except Exception:
        pass

    return ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "28"]


def convert_to_portrait(input_path: str, output_path: str, w: int, h: int):
    """
    Resize using letterboxing (NO CROPPING).
    """

    filter_chain = (
        f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
        f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=black"
    )

    cmd = [
        "ffmpeg",
        "-y",
        "-threads", "0",
        "-i", input_path,
        "-vf", filter_chain,
        *_hw_accel_args(),
        "-c:a", "copy",
        "-movflags", "+faststart",
        output_path,
    ]

    return subprocess.run(cmd, capture_output=True, text=True, timeout=600)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "ffmpeg": ffmpeg_available()}


@app.post("/resize-to-portrait")
async def resize_to_portrait(
    file: UploadFile = File(...),
    preset: str = Form("tiktok"),
    x_internal_token: str | None = Header(default=None),
):
    require_token(x_internal_token)

    if preset not in PORTRAIT_PRESETS:
        raise HTTPException(400, "Invalid preset")

    if not ffmpeg_available():
        raise HTTPException(500, "FFmpeg not installed")

    suffix = Path(file.filename).suffix or ".mp4"
    job_id = uuid.uuid4().hex[:8]

    tmp_in = None

    try:
        # Save temp file
        content = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(content)
            tmp_in = tmp.name

        w, h = get_dimensions(tmp_in)
        orientation = get_orientation(w, h)

        if orientation != "landscape":
            raise HTTPException(
                400,
                f"Video already {orientation} ({w}x{h}) — expected landscape"
            )

        cfg = PORTRAIT_PRESETS[preset]

        out_name = f"{job_id}_{Path(file.filename).stem}_portrait.mp4"
        output_path = str(SHARED_OUTPUT / out_name)

        t0 = time.time()
        result = convert_to_portrait(tmp_in, output_path, cfg["width"], cfg["height"])
        duration = round(time.time() - t0, 2)

        if result.returncode != 0:
            raise HTTPException(
                500,
                {
                    "error": "FFmpeg failed",
                    "stderr": result.stderr[-800:]
                }
            )

        size_mb = os.path.getsize(output_path) / (1024 ** 2)

        return {
            "status": "success",
            "job_id": job_id,
            "preset": preset,
            "input_dims": {"width": w, "height": h},
            "output_dims": {"width": cfg["width"], "height": cfg["height"]},
            "output_path": output_path,
            "size_mb": round(size_mb, 2),
            "duration_s": duration,
        }

    finally:
        if tmp_in and os.path.exists(tmp_in):
            os.unlink(tmp_in)


@app.get("/download/{filename}")
async def download(filename: str, x_internal_token: str | None = Header(default=None)):
    require_token(x_internal_token)

    file_path = SHARED_OUTPUT / filename
    if not file_path.exists():
        raise HTTPException(404, "File not found")

    return FileResponse(str(file_path), media_type="video/mp4", filename=filename)