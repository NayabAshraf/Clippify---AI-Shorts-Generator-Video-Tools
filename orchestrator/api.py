
"""
orchestrator/api.py — Pipeline Orchestrator
Port: 8000
Run: uvicorn orchestrator.api:app --port 8000
"""

import asyncio
import shutil
import time
import traceback
import uuid
from pathlib import Path

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass

import ffmpeg
import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from shared.config import FFPROBE_BIN, SERVICE_URLS, SHARED_STORAGE_ROOT
from shared.http_client import ServiceClient

app = FastAPI(title="Reel Generator — Orchestrator", version="2.0.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

VIDEO_UPLOAD_DIR = Path(SHARED_STORAGE_ROOT) / "uploads"
VIDEO_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def _get_video_dimensions(video_path: str) -> tuple[int, int]:
    """Returns (width, height) of the first video stream."""
    try:
        probe = ffmpeg.probe(video_path, cmd=FFPROBE_BIN)
    except ffmpeg.Error as exc:
        raise HTTPException(500, f"ffprobe failed: {exc.stderr.decode(errors='replace')}") from exc
    stream = next((s for s in probe["streams"] if s["codec_type"] == "video"), None)
    if not stream:
        raise HTTPException(422, "No video stream found.")
    return int(stream["width"]), int(stream["height"])


async def _service_health(name: str) -> dict:
    try:
        async with ServiceClient(name) as client:
            ok = await client.health()
        return {"service": name, "status": "ok" if ok else "degraded", "url": SERVICE_URLS[name]}
    except Exception as exc:
        return {"service": name, "status": "unreachable", "error": str(exc), "url": SERVICE_URLS.get(name)}


def _safe_unlink(path: Path, retries: int = 6, delay: float = 0.5) -> None:
    """
    Retry-delete for Windows — WinError 32 happens when a subprocess (ffprobe/
    ffmpeg) still holds the file handle for a fraction of a second after exit.
    """
    for attempt in range(retries):
        try:
            path.unlink(missing_ok=True)
            return
        except PermissionError:
            if attempt < retries - 1:
                time.sleep(delay)


@app.post("/api/v1/generate")
async def generate(
    file:         UploadFile = File(...),
    preset:       str        = Form("tiktok"),
    output_width: int        = Form(1080),
    keep_audio:   bool       = Form(False),
) -> JSONResponse:

    job_id = uuid.uuid4().hex[:8]
    timing: dict[str, float] = {}

    suffix     = Path(file.filename or "video.mp4").suffix.lower() or ".mp4"
    video_path = VIDEO_UPLOAD_DIR / f"{job_id}_source{suffix}"

    try:
        with open(video_path, "wb") as fh:
            shutil.copyfileobj(file.file, fh)
    except Exception as exc:
        raise HTTPException(500, f"Failed to save upload: {exc}") from exc
    finally:
        await file.close()

    src_w, src_h = _get_video_dimensions(str(video_path))

    try:
        # ── Stage 1: Audio extraction ─────────────────────────────────────
        t0 = time.time()
        with open(video_path, "rb") as video_fh:
            async with ServiceClient("audio_extractor") as client:
                audio_resp = await client.post(
                    "/extract-audio",
                    files={"file": (video_path.name, video_fh, "video/mp4")},
                )
        audio_data = audio_resp.json()
        audio_path = audio_data["audio_path"]
        timing["audio_extraction_s"] = round(time.time() - t0, 2)

        # ── Stage 2: Transcription ────────────────────────────────────────
        t0 = time.time()
        with open(audio_path, "rb") as audio_fh:
            async with ServiceClient("transcriber") as client:
                resp = await client.post(
                    "/transcribe",
                    files={"file": (Path(audio_path).name, audio_fh, "audio/mp3")},
                    data={"include_timestamps": "true"},
                )
                transcript_resp = resp.json()

        transcript = {
            "text":     transcript_resp.get("transcript", ""),
            "segments": transcript_resp.get("segments", []),
        }
        timing["transcription_s"] = round(time.time() - t0, 2)

        # ── Stage 3: Clip analysis ────────────────────────────────────────
        t0 = time.time()
        async with ServiceClient("clip_analyzer") as client:
            analysis_resp = await client.post_json(
                "/analyze", json={"transcript": transcript, "video_duration": None}
            )
        segments = analysis_resp["segments"]
        timing["analysis_s"] = round(time.time() - t0, 2)

        # ── Stage 4: Clip extraction ──────────────────────────────────────
        t0 = time.time()
        async with ServiceClient("clip_extractor") as client:
            extraction_resp = await client.post_json(
                "/extract-clips",
                json={
                    "source_video_path": str(video_path),
                    "segments":          segments,
                    "preset":            preset,
                    "output_width":      output_width,
                    "job_id":            job_id,
                },
            )
        clips      = extraction_resp["clips"]
        clip_paths = [c["clip_path"] for c in clips]
        timing["extraction_s"] = round(time.time() - t0, 2)

        # ── Stage 5: Subtitle burning ─────────────────────────────────────
        t0 = time.time()
        async with ServiceClient("subtitle_burner") as client:
            subtitle_resp = await client.post_json(
                "/burn-subtitles", json={"clip_paths": clip_paths, "job_id": job_id}
            )
        subtitled    = subtitle_resp["subtitled"]
        subtitle_map = {s["raw_clip"]: s["subtitled_clip"] for s in subtitled}
        timing["subtitle_s"] = round(time.time() - t0, 2)

        seg_map     = {i + 1: seg for i, seg in enumerate(segments)}
        final_clips = [
            {
                "index":          clip["index"],
                "title":          seg_map.get(clip["index"], {}).get("title", clip["title"]),
                "viral_score":    seg_map.get(clip["index"], {}).get("viral_score"),
                "reason":         seg_map.get(clip["index"], {}).get("reason"),
                "start":          clip["start"],
                "end":            clip["end"],
                "canvas":         clip["canvas"],
                "raw_clip":       clip["clip_path"],
                "subtitled_clip": subtitle_map.get(clip["clip_path"]),
            }
            for clip in clips
        ]

        return JSONResponse({
            "status": "success",
            "job_id": job_id,
            "preset": preset,
            "source": {"width": src_w, "height": src_h},
            "clips":  final_clips,
            "timing": timing,
        })

    except HTTPException:
        raise
    except httpx.HTTPStatusError as exc:
        traceback.print_exc()
        try:
            detail = exc.response.json().get("detail", exc.response.text)
        except Exception:
            detail = exc.response.text
        raise HTTPException(502, f"Sub-service error [{exc.response.status_code}]: {detail}") from exc
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(500, f"Pipeline error: {exc}") from exc
    finally:
        if not keep_audio:
            _safe_unlink(video_path)


@app.get("/api/v1/health")
async def health() -> dict:
    return {"status": "ok", "service": "orchestrator"}


@app.get("/api/v1/services")
async def services_health() -> JSONResponse:
    results = await asyncio.gather(*[_service_health(name) for name in SERVICE_URLS])
    all_ok  = all(r["status"] == "ok" for r in results)
    return JSONResponse(
        {"orchestrator": "ok", "services": results, "all_healthy": all_ok},
        status_code=200 if all_ok else 207,
    )