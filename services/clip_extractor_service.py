"""
services/clip_extractor_service.py
=====================================
Microservice: Clip Extractor
Port: 8004

NEW FEATURE:
- Added "keep_original" preset
- Smart resizing:
    • If preset = keep_original → ALWAYS keep original
    • If preset != keep_original:
        → Apply resizing ONLY if video is landscape
        → Otherwise keep original
"""

import os
import subprocess
import time
import uuid
from pathlib import Path

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import ffmpeg
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from shared.config import FFMPEG_BIN, SHARED_STORAGE_ROOT
from shared.http_client import require_internal_token

app = FastAPI(
    title="Clip Extractor Service",
    version="2.0.0",
)

CLIPS_OUT_DIR = Path(SHARED_STORAGE_ROOT) / "clips"
CLIPS_OUT_DIR.mkdir(parents=True, exist_ok=True)

# ✅ UPDATED PRESETS
RATIO_PRESETS: dict[str, tuple[int, int] | None] = {
    "tiktok":        (9, 16),
    "instagram":     (4, 5),
    "square":        (1, 1),
    "keep_original": None,
}


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _is_landscape(video_path: str) -> bool:
    try:
        probe = ffmpeg.probe(video_path)
        stream = next(s for s in probe["streams"] if s["codec_type"] == "video")
        w, h = int(stream["width"]), int(stream["height"])
        return w > h
    except Exception:
        return True  # fallback


def _canvas(preset: str, width: int) -> tuple[int, int]:
    ratio_w, ratio_h = RATIO_PRESETS[preset]
    w = width + (width % 2)
    h = int(w * ratio_h / ratio_w)
    h = h + (h % 2)
    return w, h


def _vf(out_w: int, out_h: int) -> str:
    return (
        f"scale=w={out_w}:h={out_h}:force_original_aspect_ratio=decrease:flags=lanczos,"
        f"scale=trunc(iw/2)*2:trunc(ih/2)*2,"
        f"pad={out_w}:{out_h}:(ow-iw)/2:(oh-ih)/2:black"
    )


def _extract_one(
    source: str,
    out_path: str,
    start: float,
    duration: float,
    preset: str,
    output_width: int,
) -> bool:

    is_landscape = _is_landscape(source)

    cmd = [
        FFMPEG_BIN, "-y",
        "-ss", str(start),
        "-i", source,
        "-t", str(duration),
    ]

    # ✅ SMART LOGIC
    if preset != "keep_original" and is_landscape:
        out_w, out_h = _canvas(preset, output_width)
        cmd += [
            "-vf", _vf(out_w, out_h),
            "-c:v", "libx264", "-preset", "superfast", "-crf", "23",
        ]
    else:
        # Keep original resolution
        cmd += ["-c:v", "copy"]

    cmd += [
        "-c:a", "aac",
        "-movflags", "+faststart",
        out_path,
    ]

    result = subprocess.run(cmd, capture_output=True)
    return result.returncode == 0


# ─────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────

class Segment(BaseModel):
    start: float
    end: float
    title: str = ""


class ExtractRequest(BaseModel):
    source_video_path: str
    segments: list[Segment]
    preset: str = Field("tiktok", pattern="^(tiktok|instagram|square|keep_original)$")
    output_width: int = Field(1080, ge=480, le=3840)
    job_id: str | None = None


# ─────────────────────────────────────────────
# Endpoint
# ─────────────────────────────────────────────

@app.post(
    "/extract-clips",
    dependencies=[Depends(require_internal_token)],
)
async def extract_clips(req: ExtractRequest) -> JSONResponse:

    source = Path(req.source_video_path)

    if not source.exists():
        raise HTTPException(404, f"Source video not found: {source}")

    if req.preset not in RATIO_PRESETS:
        raise HTTPException(422, f"Invalid preset '{req.preset}'")

    job_id = req.job_id or uuid.uuid4().hex[:8]
    job_dir = CLIPS_OUT_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    created = []
    t0 = time.time()

    is_landscape = _is_landscape(str(source))

    for i, seg in enumerate(req.segments, 1):

        duration = seg.end - seg.start
        if duration <= 0:
            continue

        safe_title = seg.title.replace(" ", "_").replace("/", "-")[:40]
        filename = f"clip_{i:02d}_{safe_title}.mp4"
        out_path = str(job_dir / filename)

        ok = _extract_one(
            source=str(source),
            out_path=out_path,
            start=seg.start,
            duration=duration,
            preset=req.preset,
            output_width=req.output_width,
        )

        if ok:
            # ✅ Canvas logic
            if req.preset == "keep_original" or not is_landscape:
                canvas = "original"
            else:
                out_w, out_h = _canvas(req.preset, req.output_width)
                canvas = f"{out_w}x{out_h}"

            created.append({
                "index": i,
                "title": seg.title,
                "clip_path": out_path,
                "start": seg.start,
                "end": seg.end,
                "canvas": canvas,
            })

    return JSONResponse({
        "job_id": job_id,
        "clips": created,
        "count": len(created),
        "duration_s": round(time.time() - t0, 2),
    })


@app.get("/health")
async def health():
    return {"status": "ok"}