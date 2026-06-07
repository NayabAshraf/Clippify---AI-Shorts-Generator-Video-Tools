"""
services/clip_analyzer_service.py
Port: 8003
Run: uvicorn services.clip_analyzer_service:app --port 8003
"""

import json
import shutil
import tempfile
import time
from pathlib import Path
from typing import Optional

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import ffmpeg
from fastapi import Depends, FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ✅ Removed GROQ_API_KEY import
from shared.config import FFPROBE_BIN, SHARED_STORAGE_ROOT
from shared.http_client import require_internal_token, ServiceClient

app = FastAPI(title="Clip Analyzer Service", version="2.0.0")

# Create temp directory for uploads
TEMP_UPLOAD_DIR = Path(SHARED_STORAGE_ROOT) / "temp_uploads"
TEMP_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

_groq_client = None


# ✅ Groq API key 
def _get_groq():
    global _groq_client
    if _groq_client is None:
        key = " "   # Add your Groq API key here

        if not key:
            raise RuntimeError("GROQ_API_KEY is missing.")

        from groq import Groq
        _groq_client = Groq(api_key=key)
    return _groq_client


SYSTEM_PROMPT = """You are an expert viral short-form content editor.
Given a video transcript with word-level timestamps, identify the 3-5 MOST IMPORTANT and CONTEXTUALLY COMPLETE segments (30-120 seconds each) that deliver a full message with high viral potential for TikTok, YouTube Shorts, or Instagram Reels.

MANDATORY CRITERIA for EVERY segment (reject if not met):
- Strong hook in first 3 seconds (surprising, emotional, controversial, curiosity-inducing)
- FULL CONTEXT: Self-contained narrative arc (setup + key point + payoff/resolution) — stands alone without prior video context
- Tied to video's core theme/message — no random filler, chit-chat, or gibberish
- Emotional peaks: laughter, shock, inspiration, tension, relatability
- Punchy pacing — no long pauses; delivers complete, meaningful takeaway

Return ONLY a valid JSON array. No explanation, no markdown. Example format:
[
  {
    "start": 12.4,
    "end": 54.1,
    "title": "The Shocking Reveal",
    "reason": "Complete arc: Curiosity hook → evidence → emotional payoff; directly advances core thesis with high shareability",
    "viral_score": 92,
    "important_text": "This is the actual spoken text from this segment that makes it viral"
  }
]"""

SYSTEM_PROMPT_TEXT_ONLY = """You are an expert content analyst. Given the following text transcript from a video or audio, extract the 3-5 MOST IMPORTANT and CONTEXTUALLY COMPLETE segments that deliver a full message and would work as standalone short-form content.

MANDATORY CRITERIA for EVERY segment (reject if not met):
- Delivers complete narrative arc (setup + key point + payoff/resolution)
- Tied to core theme — no random chit-chat, filler, or gibberish sentences
- Engaging hook + meaningful takeaway suitable for social media

For each segment, provide:
- A compelling title
- A brief reason why this segment is important/engaging (must mention contextual completeness)
- A viral score (0-100)
- The important text/key quote from this segment (full excerpt showing complete message)

Return ONLY a valid JSON array. No explanation, no markdown. Example format:
[
  {
    "start": 0,
    "end": 0,
    "title": "Key Insight",
    "reason": "Complete arc: Problem setup → evidence → actionable solution; resonates as standalone viral clip",
    "viral_score": 85,
    "important_text": "The actual important text or quote from the content showing full context"
  }
]

Note: Since there are no timestamps for text-only input, use start=0 and end=0."""

IMPORTANT_TEXT_PROMPT = """You are an expert content analyst. Given the full video transcript with timestamps, extract ONLY the MOST RELEVANT and CONTEXTUALLY COMPLETE key moments from the entire content — moments that deliver a full idea/message tied to the video's core theme.

MANDATORY CRITERIA for EVERY moment (reject random talk/gibberish):
- Full context: Complete thought arc (setup + key point + payoff) — not isolated sentences
- Directly advances main arguments/theme — main points, insights, data, quotes only
- Meaningful standalone value (could work as short social clip)

For each important moment, provide:
- The exact start and end timestamps
- The actual important text/speech from that moment
- A brief reason why this text is important/noteworthy
- A relevance score (0-100)

Return ONLY a valid JSON array. No explanation, no markdown.
"""


def _flatten_transcript(transcript: dict) -> str:
    lines = []
    for seg in transcript.get("segments", []):
        lines.append(f"[{seg['start']:.2f}s - {seg['end']:.2f}s] {seg.get('text','').strip()}")
    return "\n".join(lines)


def _get_audio_duration(audio_path: str) -> float:
    try:
        probe = ffmpeg.probe(audio_path, cmd=FFPROBE_BIN)
        return float(probe["format"]["duration"])
    except Exception:
        return 0.0


def _cleanup_temp_file(file_path: Path):
    try:
        if file_path.exists():
            file_path.unlink()
    except Exception:
        pass


class AnalyzeRequest(BaseModel):
    transcript: dict
    video_duration: float | None = None
    extract_important_text: bool = False


@app.post("/analyze", dependencies=[Depends(require_internal_token)])
async def analyze(req: AnalyzeRequest) -> JSONResponse:
    transcript_text = _flatten_transcript(req.transcript)

    if not transcript_text.strip():
        raise HTTPException(422, "Transcript is empty.")

    user_msg = transcript_text
    if req.video_duration:
        user_msg = f"[Total duration: {req.video_duration:.1f}s]\n\n" + user_msg

    t0 = time.time()
    groq = _get_groq()

    response = groq.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        temperature=0.3,
        max_tokens=1024,
    )

    elapsed = time.time() - t0
    raw = response.choices[0].message.content.strip()

    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    try:
        segments: list[dict] = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(502, f"LLM returned invalid JSON: {exc}") from exc

    segments.sort(key=lambda x: x.get("viral_score", 0), reverse=True)

    return JSONResponse({
        "segments": segments,
        "count": len(segments),
        "duration_s": round(elapsed, 2),
    })


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "service": "clip_analyzer"}


import uuid