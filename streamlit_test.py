"""
streamlit_test.py — Microservices Test Dashboard
==================================================
Tests every microservice endpoint individually AND the full pipeline.

Run:
    streamlit run streamlit_test.py

Requirements:
    pip install streamlit httpx opencv-python
"""

import json
import os
import tempfile
import time
from pathlib import Path

import cv2
import httpx
import streamlit as st

import asyncio
import logging

logging.getLogger("asyncio").setLevel(logging.CRITICAL)
# ── Config ────────────────────────────────────────────────────────────────────

SERVICES = {
    "orchestrator":    "http://localhost:8000",
    "audio_extractor": "http://localhost:8001",
    "transcriber":     "http://localhost:8002",
    "clip_analyzer":   "http://localhost:8003",
    "clip_extractor":  "http://localhost:8004",
    "subtitle_burner": "http://localhost:8005",
    "resizer":         "http://localhost:8006",          # ← NEW
}

INTERNAL_TOKEN = os.getenv("INTERNAL_TOKEN", "dev-secret-change-me")
HEADERS        = {"X-Internal-Token": INTERNAL_TOKEN}
TIMEOUT        = 1500   # seconds

VALID_PRESETS  = {"tiktok", "instagram", "square", "keep_original"}
VALID_WIDTHS   = [720, 1080, 1280]

PRESET_OPTIONS = {
    "tiktok":        "TikTok / Reels  (9:16)",
    "instagram":     "Instagram Feed  (4:5)",
    "square":        "Square          (1:1)",
    "keep_original": "Keep Original   (No Resize)",
}

PORTRAIT_PRESET_OPTIONS = {
    "tiktok":    "TikTok / Reels  (9:16  —  1080×1920)",
    "instagram": "Instagram Feed  (4:5   —  1080×1350)",
    "square":    "Square          (1:1   —  1080×1080)",
}

# Presets that actually resize the video
RESIZE_PRESETS = {"tiktok", "instagram", "square"}

ALLOWED_VIDEO_TYPES = ["mp4", "mov", "avi", "mkv"]
ALLOWED_AUDIO_TYPES = ["mp3", "wav", "m4a", "mp4"]

# ── Page setup ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Clippify — AI Video Tool",
    page_icon="🧪",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Sora:wght@300;500;700&display=swap');

html, body, [class*="css"] { font-family: 'Sora', sans-serif; }
h1, h2, h3 { font-family: 'Sora', sans-serif; font-weight: 700; }

.main { background: #0a0e1a; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

.pill-ok   { display:inline-block; background:#0d3b2e; color:#34d399;
             border:1px solid #059669; border-radius:20px;
             padding:2px 12px; font-size:12px; font-family:'JetBrains Mono',monospace; }
.pill-fail { display:inline-block; background:#3b0d0d; color:#f87171;
             border:1px solid #dc2626; border-radius:20px;
             padding:2px 12px; font-size:12px; font-family:'JetBrains Mono',monospace; }
.pill-warn { display:inline-block; background:#3b2a0d; color:#fbbf24;
             border:1px solid #d97706; border-radius:20px;
             padding:2px 12px; font-size:12px; font-family:'JetBrains Mono',monospace; }

.json-box {
    background: #0d1117; border: 1px solid #21262d;
    border-radius: 8px; padding: 14px 18px;
    font-family: 'JetBrains Mono', monospace; font-size: 12px;
    color: #79c0ff; overflow-x: auto; white-space: pre-wrap;
    max-height: 340px; overflow-y: auto;
}

.timing-row { display:flex; align-items:center; gap:10px; margin:4px 0; font-size:13px; }
.timing-bar-bg { flex:1; background:#1c2130; border-radius:4px; height:8px; }
.timing-bar-fill { height:8px; border-radius:4px; background:linear-gradient(90deg,#6366f1,#818cf8); }
.timing-label { min-width:160px; color:#94a3b8; font-family:'JetBrains Mono',monospace; font-size:12px; }
.timing-val   { min-width:48px; text-align:right; color:#c4b5fd;
                font-family:'JetBrains Mono',monospace; font-size:12px; }

button[data-baseweb="tab"] { font-family:'Sora',sans-serif !important; font-size:14px !important; }

.val-error {
    background: #3b0d0d; border: 1px solid #dc2626; border-radius: 8px;
    padding: 10px 14px; font-size: 13px; color: #f87171;
    margin-bottom: 12px; font-family: 'JetBrains Mono', monospace;
}
.val-warn {
    background: #3b2a0d; border: 1px solid #d97706; border-radius: 8px;
    padding: 10px 14px; font-size: 13px; color: #fbbf24;
    margin-bottom: 12px; font-family: 'JetBrains Mono', monospace;
}

/* ── Resizer-specific styles ───────────────────────────────────────────── */
.orientation-badge-landscape {
    display: inline-flex; align-items: center; gap: 8px;
    background: #0d2b3e; color: #38bdf8;
    border: 1px solid #0284c7; border-radius: 10px;
    padding: 8px 16px; font-size: 13px;
    font-family: 'JetBrains Mono', monospace;
    margin: 8px 0;
}
.orientation-badge-portrait {
    display: inline-flex; align-items: center; gap: 8px;
    background: #3b0d0d; color: #f87171;
    border: 1px solid #dc2626; border-radius: 10px;
    padding: 8px 16px; font-size: 13px;
    font-family: 'JetBrains Mono', monospace;
    margin: 8px 0;
}
.orientation-badge-square {
    display: inline-flex; align-items: center; gap: 8px;
    background: #3b2a0d; color: #fbbf24;
    border: 1px solid #d97706; border-radius: 10px;
    padding: 8px 16px; font-size: 13px;
    font-family: 'JetBrains Mono', monospace;
    margin: 8px 0;
}
.resize-alert {
    background: linear-gradient(135deg, #1e0a0a 0%, #2d0f0f 100%);
    border: 1px solid #dc2626; border-left: 4px solid #ef4444;
    border-radius: 10px; padding: 16px 20px;
    color: #fca5a5; font-size: 14px; margin: 12px 0;
}
.resize-alert-title {
    font-weight: 700; font-size: 15px; color: #f87171;
    margin-bottom: 6px; font-family: 'Sora', sans-serif;
}
.resize-success-box {
    background: linear-gradient(135deg, #0a1e10 0%, #0f2d18 100%);
    border: 1px solid #16a34a; border-left: 4px solid #22c55e;
    border-radius: 10px; padding: 16px 20px;
    color: #86efac; font-size: 14px; margin: 12px 0;
}
.dims-card {
    background: #0d1117; border: 1px solid #21262d; border-radius: 8px;
    padding: 12px 16px; text-align: center;
}
.dims-card-label { font-size: 11px; color: #64748b; font-family: 'JetBrains Mono', monospace; margin-bottom: 4px; }
.dims-card-value { font-size: 18px; font-weight: 700; color: #e2e8f0; font-family: 'JetBrains Mono', monospace; }
.arrow-between { font-size: 28px; color: #6366f1; text-align: center; padding-top: 10px; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────

st.markdown("""
<h1 style='margin-bottom:0'>🧪 Clippify - AI Video Tool</h1>
<p style='color:#64748b; margin-top:4px; font-size:15px;'>
Clip, caption, resize — done for you in minutes.
</p>
""", unsafe_allow_html=True)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# CACHED HELPERS
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def get_video_orientation(file_bytes: bytes, filename: str) -> str | None:
    """
    Returns 'landscape', 'portrait', 'square', or None.
    Cached on (file_bytes, filename) — changing the file re-runs automatically;
    toggling any other widget does NOT re-run this.
    """
    suffix   = Path(filename).suffix.lower() or ".mp4"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        cap    = cv2.VideoCapture(tmp_path)
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if width <= 0 or height <= 0:
            return None
        if width > height:
            return "landscape"
        if height > width:
            return "portrait"
        return "square"
    except Exception:
        return None
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


@st.cache_data(show_spinner=False)
def get_video_dimensions(file_bytes: bytes, filename: str) -> tuple[int, int] | None:
    """Returns (width, height) or None on failure. Cached per file."""
    suffix   = Path(filename).suffix.lower() or ".mp4"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        cap    = cv2.VideoCapture(tmp_path)
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if width <= 0 or height <= 0:
            return None
        return width, height
    except Exception:
        return None
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


@st.cache_data(ttl=10, show_spinner=False)
def ping_all_services() -> dict[str, tuple[bool, str]]:
    """
    Pings every service once and caches the result for 10 seconds.
    """
    results = {}
    for name, url in SERVICES.items():
        path = "/api/v1/health" if name == "orchestrator" else "/health"
        try:
            r = httpx.get(f"{url}{path}", timeout=4)
            if r.status_code == 200 and r.json().get("status") == "ok":
                results[name] = (True, "ok")
            else:
                results[name] = (False, f"HTTP {r.status_code}")
        except Exception as exc:
            results[name] = (False, str(exc)[:60])
    return results


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def validate_preset(preset: str) -> str | None:
    if preset not in VALID_PRESETS:
        return f"Unknown preset '{preset}'. Choose from: {sorted(VALID_PRESETS)}"
    return None


def validate_output_width(width: int, preset: str = "") -> str | None:
    if preset == "keep_original":
        return None
    if width not in VALID_WIDTHS:
        return f"Invalid output width {width}px. Choose from: {VALID_WIDTHS}"
    return None


def validate_video_file(uploaded_file) -> list[str]:
    errors = []
    if uploaded_file is None:
        errors.append("No file uploaded. Please select a video.")
        return errors
    ext = Path(uploaded_file.name).suffix.lstrip(".").lower()
    if ext not in ALLOWED_VIDEO_TYPES:
        errors.append(
            f"Unsupported file type '.{ext}'. "
            f"Allowed: {', '.join('.' + t for t in ALLOWED_VIDEO_TYPES)}"
        )
    size_mb = len(uploaded_file.getvalue()) / (1024 ** 2)
    if size_mb > 2048:
        errors.append(f"File is {size_mb:.0f} MB — exceeds 2 GB limit.")
    return errors


def validate_audio_file(uploaded_file) -> list[str]:
    errors = []
    if uploaded_file is None:
        errors.append("No audio file uploaded.")
        return errors
    ext = Path(uploaded_file.name).suffix.lstrip(".").lower()
    if ext not in ALLOWED_AUDIO_TYPES:
        errors.append(
            f"Unsupported audio type '.{ext}'. "
            f"Allowed: {', '.join('.' + t for t in ALLOWED_AUDIO_TYPES)}"
        )
    return errors


def validate_transcript_json(raw: str) -> tuple[dict | None, str | None]:
    if not raw.strip():
        return None, "Transcript JSON is empty."
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as exc:
        return None, f"Invalid JSON: {exc}"
    if not isinstance(obj, dict):
        return None, "Transcript must be a JSON object, not an array."
    if "text" not in obj:
        return None, "Transcript JSON is missing the required 'text' field."
    if not isinstance(obj.get("text", ""), str):
        return None, "'text' field must be a string."
    if "segments" in obj and not isinstance(obj["segments"], list):
        return None, "'segments' field must be an array."
    if not obj.get("text", "").strip():
        return None, (
            "Transcript 'text' is empty — no speech detected. "
            "The analyzer requires spoken content to find viral segments."
        )
    return obj, None


def validate_segments_json(raw: str) -> tuple[list | None, str | None]:
    if not raw.strip():
        return None, "Segments JSON is empty."
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as exc:
        return None, f"Invalid JSON: {exc}"
    if not isinstance(obj, list):
        return None, "Segments must be a JSON array, e.g. [{...}, {...}]."
    if len(obj) == 0:
        return None, "Segments list is empty. The clip extractor needs at least one segment."
    for i, seg in enumerate(obj):
        if not isinstance(seg, dict):
            return None, f"Segment #{i + 1} must be a JSON object."
        for required in ("start", "end"):
            if required not in seg:
                return None, f"Segment #{i + 1} is missing required field '{required}'."
        try:
            start, end = float(seg["start"]), float(seg["end"])
        except (TypeError, ValueError):
            return None, f"Segment #{i + 1} 'start'/'end' must be numbers."
        if start < 0:
            return None, f"Segment #{i + 1} has negative 'start' value ({start})."
        if end <= start:
            return None, f"Segment #{i + 1} 'end' ({end}) must be greater than 'start' ({start})."
    return obj, None


def validate_source_video_path(path: str) -> str | None:
    if not path.strip():
        return "Source video path is required."
    if not Path(path).is_absolute():
        return f"Path '{path}' looks relative — provide the full absolute path on shared storage."
    return None


def validate_clip_paths(raw_text: str) -> tuple[list[str] | None, str | None]:
    paths = [p.strip() for p in raw_text.strip().splitlines() if p.strip()]
    if not paths:
        return None, "No clip paths provided. Enter at least one path."
    for p in paths:
        if not Path(p).is_absolute():
            return None, f"Path '{p}' looks relative — provide full absolute paths on shared storage."
    return paths, None


# ─────────────────────────────────────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def show_errors(errors: list[str] | str) -> None:
    if isinstance(errors, str):
        errors = [errors]
    for err in errors:
        st.markdown(f"<div class='val-error'>⛔ {err}</div>", unsafe_allow_html=True)


def show_warning(msg: str) -> None:
    st.markdown(f"<div class='val-warn'>⚠️ {msg}</div>", unsafe_allow_html=True)


def render_json(data: dict | list | str) -> None:
    text = json.dumps(data, indent=2) if not isinstance(data, str) else data
    st.markdown(f"<div class='json-box'>{text}</div>", unsafe_allow_html=True)


def render_timing(timing: dict) -> None:
    if not timing:
        return
    max_val = max(timing.values()) or 1
    st.markdown("**Per-stage timing**")
    for key, val in timing.items():
        pct = int(val / max_val * 100)
        st.markdown(
            f"<div class='timing-row'>"
            f"<span class='timing-label'>{key}</span>"
            f"<div class='timing-bar-bg'><div class='timing-bar-fill' style='width:{pct}%'></div></div>"
            f"<span class='timing-val'>{val:.1f}s</span>"
            f"</div>",
            unsafe_allow_html=True,
        )


def render_orientation_badge(orientation: str, w: int, h: int) -> None:
    icons = {"landscape": "📺", "portrait": "📱", "square": "⬛"}
    cls   = {
        "landscape": "orientation-badge-landscape",
        "portrait":  "orientation-badge-portrait",
        "square":    "orientation-badge-square",
    }.get(orientation, "orientation-badge-square")
    icon = icons.get(orientation, "🎬")
    st.markdown(
        f"<div class='{cls}'>"
        f"{icon} &nbsp; <strong>{orientation.upper()}</strong> &nbsp; "
        f"<span style='opacity:0.7'>{w} × {h} px</span>"
        f"</div>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — Service health panel
# ─────────────────────────────────────────────────────────────────────────────

with st.expander("**Service Health**  — click to expand / refresh", expanded=False):
    if st.button("🔄 Refresh health", key="refresh_health"):
        ping_all_services.clear()
        st.rerun()

    health_results = ping_all_services()
    cols = st.columns(len(SERVICES))
    for col, (name, url) in zip(cols, SERVICES.items()):
        ok, msg   = health_results[name]
        pill_cls  = "pill-ok"  if ok else "pill-fail"
        pill_text = "● online" if ok else f"✕ {msg}"
        with col:
            st.markdown(f"**{name.replace('_', ' ').title()}**")
            st.markdown(f"`{url}`")
            st.markdown(f"<span class='{pill_cls}'>{pill_text}</span>", unsafe_allow_html=True)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────

(
    tab_pipeline, tab_audio, tab_transcribe,
    tab_analyze, tab_subtitle,
    tab_resizer,
) = st.tabs([
    "🚀 AI Shorts Generator",
    "🎧 Audio Extractor",
    "📝 Transcriber",
    "🧠 Clip Analyzer",
    "🎬 Subtitle Burner",
    "🔄 Video Resizer",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Full Pipeline
# ─────────────────────────────────────────────────────────────────────────────

with tab_pipeline:
    st.markdown("### Full pipeline via Orchestrator")
    st.caption("Calls `POST /api/v1/generate` — runs all 5 stages in sequence.")

    col_l, col_r = st.columns([1, 1])

    with col_l:
        uploaded = st.file_uploader(
            "Upload video",
            type=ALLOWED_VIDEO_TYPES,
            key="pipeline_upload",
        )

        orientation = None
        if uploaded:
            orientation = get_video_orientation(uploaded.getvalue(), uploaded.name)

        is_non_landscape = orientation in ("portrait", "square")

        forced_preset = "keep_original" if is_non_landscape else None
        preset_keys   = list(PRESET_OPTIONS.keys())
        default_index = preset_keys.index(
            forced_preset if forced_preset
            else st.session_state.get("pipeline_preset_user", "tiktok")
        )

        preset = st.selectbox(
            "Preset",
            options=preset_keys,
            format_func=lambda k: PRESET_OPTIONS[k],
            index=default_index,
            disabled=is_non_landscape,
            key="pipeline_preset",
        )
        if not is_non_landscape:
            st.session_state["pipeline_preset_user"] = preset

        if preset in RESIZE_PRESETS:
            output_width = st.select_slider(
                "Output width (px)",
                options=VALID_WIDTHS,
                value=1080,
                key="pipeline_width",
            )
        else:
            output_width = 0
            st.caption("ℹ️ Output width is not applicable for Keep Original.")

        keep_audio = st.checkbox(
            "Keep source audio after processing",
            value=False,
            key="pipeline_keep_audio",
        )

        pipeline_errors: list[str] = []
        file_errors = validate_video_file(uploaded)
        pipeline_errors.extend(file_errors)
        for err in [validate_preset(preset), validate_output_width(output_width, preset)]:
            if err:
                pipeline_errors.append(err)

        if uploaded and not file_errors:
            size_mb = len(uploaded.getvalue()) / (1024 ** 2)
            st.caption(f"📄 `{uploaded.name}` — {size_mb:.1f} MB")
            if orientation:
                st.caption(f"📐 Detected: **{orientation.upper()}**")
            if is_non_landscape:
                show_warning(
                    "Your video will not be resized for this aspect ratio and will remain "
                    "in its original aspect ratio. "
                    "For resizing to portrait format, upload in landscape format."
                )

        run_btn = st.button(
            "▶ Run Full Pipeline",
            disabled=(uploaded is None or bool(pipeline_errors)),
            key="pipeline_run",
            type="primary",
        )

    with col_r:
        if pipeline_errors:
            show_errors(pipeline_errors)
        elif uploaded is None:
            st.info("Upload a video on the left to begin.")

    if run_btn and uploaded and not pipeline_errors:
        st.markdown("---")
        prog   = st.progress(0, text="Sending to orchestrator…")
        status = st.empty()

        for pct, msg in [
            (15, "① Extracting audio…"),
            (30, "② Transcribing speech…"),
            (50, "③ Analysing viral segments…"),
            (70, "④ Extracting & resizing clips…"),
            (88, "⑤ Burning karaoke subtitles…"),
        ]:
            status.info(msg)
            prog.progress(pct, text=msg)
            time.sleep(0.3)

        try:
            with httpx.Client(timeout=TIMEOUT) as client:
                resp = client.post(
                    f"{SERVICES['orchestrator']}/api/v1/generate",
                    files={"file": (uploaded.name, uploaded.getvalue(), "video/mp4")},
                    data={
                        "preset":       preset,
                        "output_width": str(output_width),
                        "keep_audio":   str(keep_audio).lower(),
                    },
                )
                resp.raise_for_status()
                data = resp.json()

            prog.progress(100, text="Done!")
            status.success("✅ Pipeline complete!")
            st.markdown(f"**Job ID:** `{data.get('job_id')}`  |  **Preset:** `{data.get('preset')}`")
            render_timing(data.get("timing", {}))

            clips = data.get("clips", [])
            if not clips:
                st.warning("No clips were returned. The transcript may have been too short.")
            else:
                st.markdown(f"**{len(clips)} clip(s) generated:**")

            for clip in clips:
                with st.container(border=True):
                    score = clip.get("viral_score", "—")
                    st.markdown(
                        f"**#{clip['index']} — {clip.get('title', 'Clip')}**"
                        f"  &nbsp; score `{score}`"
                    )
                    st.caption(clip.get("reason", ""))
                    sub = clip.get("subtitled_clip")
                    raw = clip.get("raw_clip")
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("Subtitled clip")
                        if sub and Path(sub).exists():
                            st.video(sub)
                            with open(sub, "rb") as f:
                                st.download_button("⬇ Download subtitled", f,
                                    file_name=Path(sub).name, mime="video/mp4",
                                    key=f"dl_sub_{clip['index']}")
                        else:
                            st.warning(f"Not found:\n`{sub}`")
                    with c2:
                        st.markdown("Raw clip")
                        if raw and Path(raw).exists():
                            st.video(raw)
                            with open(raw, "rb") as f:
                                st.download_button("⬇ Download raw", f,
                                    file_name=Path(raw).name, mime="video/mp4",
                                    key=f"dl_raw_{clip['index']}")
                        else:
                            st.warning(f"Not found:\n`{raw}`")

            with st.expander("Raw JSON response"):
                render_json(data)

        except httpx.HTTPStatusError as exc:
            prog.empty()
            try:
                detail = exc.response.json().get("detail", exc.response.text)
            except Exception:
                detail = exc.response.text
            st.error(f"HTTP {exc.response.status_code}: {detail}")
        except Exception as exc:
            prog.empty()
            st.error(f"Error: {exc}")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Audio Extractor
# ─────────────────────────────────────────────────────────────────────────────

with tab_audio:
    st.markdown("### Audio Extractor  —  `POST /extract-audio`")
    st.caption("Uploads a video and extracts mono 16kHz MP3 to shared storage.")

    audio_upload = st.file_uploader("Upload video", type=ALLOWED_VIDEO_TYPES, key="audio_upload")

    audio_tab_errors = validate_video_file(audio_upload)
    if audio_tab_errors:
        show_errors(audio_tab_errors)
    elif audio_upload:
        size_mb = len(audio_upload.getvalue()) / (1024 ** 2)
        st.caption(f"📄 `{audio_upload.name}` — {size_mb:.1f} MB")
        ao = get_video_orientation(audio_upload.getvalue(), audio_upload.name)
        if ao in ("portrait", "square"):
            show_warning(
                "Your video will not be resized for this aspect ratio and will remain "
                "in its original aspect ratio. "
                "For resizing to portrait format, upload in landscape format."
            )

    if st.button("▶ Extract Audio",
                 disabled=(audio_upload is None or bool(audio_tab_errors)),
                 key="audio_run"):
        with st.spinner("Extracting…"):
            try:
                with httpx.Client(timeout=TIMEOUT) as client:
                    resp = client.post(
                        f"{SERVICES['audio_extractor']}/extract-audio",
                        files={"file": (audio_upload.name, audio_upload.getvalue(), "video/mp4")},
                        headers=HEADERS,
                    )
                    resp.raise_for_status()
                    data = resp.json()

                st.success(f"✅ Done in {data.get('duration_s')}s")
                st.markdown(f"**Audio path:** `{data.get('audio_path')}`")
                st.markdown(f"**Job ID:** `{data.get('job_id')}`")
                st.session_state["last_audio_path"] = data.get("audio_path")
                st.session_state["last_job_id"]     = data.get("job_id")

                with st.expander("Full response"):
                    render_json(data)

            except httpx.HTTPStatusError as exc:
                st.error(f"HTTP {exc.response.status_code}: {exc.response.text[:400]}")
            except Exception as exc:
                st.error(str(exc))

    if st.session_state.get("last_audio_path"):
        st.info(
            f"Last extracted audio: `{st.session_state['last_audio_path']}`\n\n"
            "Switch to the **Transcriber** tab to transcribe it."
        )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Transcriber
# ─────────────────────────────────────────────────────────────────────────────

with tab_transcribe:
    st.markdown("### Transcriber  —  `POST /transcribe`")
    st.caption("Upload an audio/video file to transcribe using Whisper.")

    audio_file = st.file_uploader("Upload audio file", type=ALLOWED_AUDIO_TYPES,
                                   key="transcribe_upload")
    include_timestamps = st.checkbox("Include word timestamps", value=True,
                                      key="transcribe_timestamps")

    transcribe_errors = validate_audio_file(audio_file) if audio_file else []
    if transcribe_errors:
        show_errors(transcribe_errors)
    elif audio_file:
        size_mb = len(audio_file.getvalue()) / (1024 ** 2)
        st.caption(f"🎵 `{audio_file.name}` — {size_mb:.1f} MB")

    if st.button("▶ Transcribe",
                 disabled=(not audio_file or bool(transcribe_errors)),
                 key="transcribe_run"):
        with st.spinner("Transcribing (this may take a minute)…"):
            try:
                with httpx.Client(timeout=TIMEOUT) as client:
                    resp = client.post(
                        f"{SERVICES['transcriber']}/transcribe",
                        files={"file": (audio_file.name, audio_file.getvalue(), "audio/mp3")},
                        data={"include_timestamps": str(include_timestamps).lower()},
                        headers=HEADERS,
                    )
                    resp.raise_for_status()
                    data = resp.json()

                transcript_text = data.get("transcript", "")
                segments        = data.get("segments", [])

                if not transcript_text.strip():
                    show_warning(
                        "Transcription returned empty text. "
                        "The clip analyzer requires spoken content — "
                        "check that the audio contains speech."
                    )
                else:
                    st.success("✅ Transcription completed!")
                    st.markdown(f"**Segments:** {len(segments)}")
                    st.markdown("**Full text:**")
                    st.info(transcript_text)
                    st.session_state["last_transcript"] = {
                        "text": transcript_text, "segments": segments
                    }
                    with st.expander("Segment details"):
                        for seg in segments[:10]:
                            st.markdown(f"`{seg['start']:.2f}s → {seg['end']:.2f}s`  {seg['text']}")

                with st.expander("Full response JSON"):
                    render_json(data)

            except httpx.HTTPStatusError as exc:
                st.error(f"HTTP {exc.response.status_code}: {exc.response.text[:400]}")
            except Exception as exc:
                st.error(str(exc))

    if st.session_state.get("last_transcript"):
        st.info("Transcript saved — switch to **Clip Analyzer** to find viral segments.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — Clip Analyzer
# ─────────────────────────────────────────────────────────────────────────────

with tab_analyze:
    st.markdown("### Clip Analyzer  —  Analyze Video/Audio or Text")
    st.caption("Upload a video/audio file or paste transcript text to identify viral segments AND important text with timestamps")

    analyze_tab1, analyze_tab2, analyze_tab3 = st.tabs([
        "📁 Upload File (Drag & Drop)",
        "📝 Paste Transcript",
        "🔤 Direct Text Analysis"
    ])

    with analyze_tab1:
        st.markdown("""
        **Upload a video or audio file** to automatically:
        1. Extract audio (if video)
        2. Transcribe speech
        3. Identify **VIRAL segments** (best for social media)
        4. Extract **IMPORTANT text with timestamps** (key insights, quotes, data points)
        """)

        uploaded_media = st.file_uploader(
            "Drag & drop or click to upload",
            type=["mp4", "mov", "avi", "mkv", "webm", "mp3", "wav", "m4a", "ogg", "flac"],
            key="analyze_file_upload",
            help="Supported formats: MP4, MOV, AVI, MKV, WebM, MP3, WAV, M4A, OGG, FLAC"
        )

        col1, col2 = st.columns(2)
        with col1:
            include_ts = st.checkbox(
                "Include word timestamps in transcription",
                value=True,
                key="analyze_include_ts",
            )
        with col2:
            extract_important = st.checkbox(
                "Extract important text with timestamps",
                value=True,
                key="extract_important_checkbox",
            )

        if uploaded_media:
            file_size_mb = len(uploaded_media.getvalue()) / (1024 ** 2)
            file_ext = Path(uploaded_media.name).suffix.lower()
            file_type = "Video" if file_ext in ['.mp4', '.mov', '.avi', '.mkv', '.webm'] else "Audio"
            col_info1, col_info2 = st.columns(2)
            with col_info1:
                st.info(f"📄 **File:** {uploaded_media.name}")
                st.info(f"📦 **Size:** {file_size_mb:.1f} MB")
            with col_info2:
                st.info(f"🎬 **Type:** {file_type}")
                st.info(f"🔤 **Format:** {file_ext.upper()}")

        analyze_file_btn = st.button(
            "🚀 Analyze File",
            disabled=uploaded_media is None,
            key="analyze_file_btn",
            type="primary",
            use_container_width=True
        )

        if analyze_file_btn and uploaded_media:
            with st.spinner("Processing file..."):
                try:
                    progress_bar = st.progress(0)
                    status_text  = st.empty()
                    steps = [
                        (15, "📤 Uploading file..."),
                        (30, "🎵 Extracting audio (if video)..."),
                        (50, "📝 Transcribing speech..."),
                        (70, "🔍 Identifying viral segments..."),
                        (85, "📖 Extracting important text..."),
                        (95, "✨ Generating insights..."),
                    ]
                    with httpx.Client(timeout=300) as client:
                        files = {"file": (uploaded_media.name, uploaded_media.getvalue(), f"{file_type.lower()}/{file_ext[1:]}")}
                        data  = {
                            "include_timestamps":       str(include_ts).lower(),
                            "extract_important_text":   str(extract_important).lower(),
                        }
                        for pct, msg in steps:
                            status_text.info(msg)
                            progress_bar.progress(pct)
                            time.sleep(0.2)
                        resp = client.post(
                            f"{SERVICES['clip_analyzer']}/analyze-from-file",
                            files=files, data=data, headers=HEADERS,
                        )
                        resp.raise_for_status()
                        result = resp.json()

                    progress_bar.progress(100)
                    status_text.empty()
                    st.success(f"✅ Analysis complete in {result.get('duration_s', 0):.1f}s!")

                    col_meta1, col_meta2, col_meta3, col_meta4 = st.columns(4)
                    with col_meta1: st.metric("File Name", result.get('file_name', 'N/A')[:30])
                    with col_meta2: st.metric("Duration", f"{result.get('duration_s', 0):.1f}s")
                    with col_meta3: st.metric("Viral Segments", result.get('viral_count', len(result.get('segments', []))))
                    with col_meta4: st.metric("Important Texts", result.get('important_count', 0))

                    with st.expander("📝 Full Transcript", expanded=False):
                        st.text_area("Complete Transcription",
                                     result.get('full_transcript', result.get('transcript', '')),
                                     height=200, disabled=True, key="analyze_transcript_display")

                    viral_segments = result.get('viral_segments', result.get('segments', []))
                    if viral_segments:
                        st.markdown(f"### 🔥 Top {len(viral_segments)} Viral Segments")
                        for idx, seg in enumerate(viral_segments, 1):
                            with st.container(border=True):
                                col_a, col_b = st.columns([3, 1])
                                with col_a:
                                    st.markdown(f"**{idx}. {seg.get('title', 'Untitled Segment')}**")
                                    if seg.get('reason'):
                                        st.caption(f"📌 {seg['reason']}")
                                    if seg.get('start', 0) > 0 or seg.get('end', 0) > 0:
                                        st.markdown(f"⏱️ `{seg['start']:.1f}s → {seg['end']:.1f}s`")
                                with col_b:
                                    score = seg.get('viral_score', 0)
                                    color = "#34d399" if score >= 80 else "#fbbf24" if score >= 60 else "#f87171"
                                    st.markdown(f"<div style='text-align:center;font-size:28px;font-weight:700;color:{color}'>{score}</div><div style='text-align:center;font-size:11px;color:#64748b'>viral score</div>", unsafe_allow_html=True)

                    important_texts = result.get('important_texts', [])
                    if important_texts and extract_important:
                        st.markdown(f"### 📖 Important Text & Key Insights ({len(important_texts)} moments)")
                        for idx, imp in enumerate(important_texts, 1):
                            with st.container(border=True):
                                col_a, col_b = st.columns([4, 1])
                                with col_a:
                                    if imp.get('start', 0) > 0 or imp.get('end', 0) > 0:
                                        st.markdown(f"**⏱️ {imp['start']:.1f}s - {imp['end']:.1f}s**")
                                    st.markdown(f"💬 \"{imp.get('important_text', imp.get('text', ''))}\"")
                                    if imp.get('reason'): st.caption(f"📌 {imp['reason']}")
                                with col_b:
                                    score = imp.get('relevance_score', imp.get('viral_score', 0))
                                    color = "#34d399" if score >= 80 else "#fbbf24" if score >= 60 else "#f87171"
                                    st.markdown(f"<div style='text-align:center;font-size:20px;font-weight:700;color:{color}'>{score}</div><div style='text-align:center;font-size:10px;color:#64748b'>relevance</div>", unsafe_allow_html=True)

                    st.download_button("💾 Download Complete Analysis (JSON)",
                                       json.dumps(result, indent=2),
                                       file_name=f"complete_analysis_{int(time.time())}.json",
                                       mime="application/json", key="analyze_download_results")

                except httpx.HTTPStatusError as exc:
                    progress_bar.empty()
                    st.error(f"HTTP Error {exc.response.status_code}")
                    try:
                        st.code(exc.response.json().get('detail', exc.response.text), language="json")
                    except Exception:
                        st.code(exc.response.text[:500])
                except Exception as exc:
                    progress_bar.empty()
                    st.error(f"Error: {str(exc)}")

    with analyze_tab2:
        st.markdown("**Paste a transcript JSON** from the Transcriber tab or any other source")
        use_last   = st.session_state.get("last_transcript") is not None
        source     = st.radio("Transcript source",
                              ["Use transcript from Transcriber tab", "Paste JSON manually"],
                              disabled=not use_last, key="analyze_source_new")
        transcript_json = ""
        if source == "Use transcript from Transcriber tab" and use_last:
            transcript_json = json.dumps(st.session_state["last_transcript"])
            st.success("Using transcript from previous step.")
        else:
            transcript_json = st.text_area("Paste transcript JSON", height=180,
                                            placeholder='{"text": "...", "segments": [...]}',
                                            key="analyze_json_new")
        video_dur = st.number_input("Video duration hint (seconds, optional)",
                                     min_value=0.0, value=0.0, key="analyze_dur_new")
        extract_important_json = st.checkbox("Also extract important text with timestamps",
                                              value=True, key="extract_important_json")
        analyze_errors: list[str] = []
        transcript_obj = None
        if transcript_json:
            transcript_obj, parse_err = validate_transcript_json(transcript_json)
            if parse_err: analyze_errors.append(parse_err)
        if analyze_errors: show_errors(analyze_errors)

        if st.button("▶ Analyze Transcript",
                     disabled=(not transcript_json or bool(analyze_errors)),
                     key="analyze_json_btn", type="primary", use_container_width=True):
            with st.spinner("Analyzing transcript with AI..."):
                try:
                    endpoint = "/analyze-with-important-text" if extract_important_json else "/analyze"
                    with httpx.Client(timeout=120) as client:
                        resp = client.post(
                            f"{SERVICES['clip_analyzer']}{endpoint}",
                            json={"transcript": transcript_obj, "video_duration": video_dur or None},
                            headers=HEADERS,
                        )
                        resp.raise_for_status()
                        data = resp.json()

                    if extract_important_json:
                        viral_segments  = data.get("viral_segments", [])
                        important_texts = data.get("important_texts", [])
                        if not viral_segments and not important_texts:
                            show_warning("No segments or important text were identified.")
                        else:
                            st.success(f"✅ Found {len(viral_segments)} viral segments and {len(important_texts)} important texts in {data.get('duration_s', 0):.1f}s")
                            if viral_segments:
                                st.markdown(f"### 🔥 Viral Segments ({len(viral_segments)})")
                                for seg in viral_segments:
                                    with st.container(border=True):
                                        col_a, col_b = st.columns([3, 1])
                                        with col_a:
                                            st.markdown(f"**{seg.get('title', '—')}**")
                                            st.caption(seg.get("reason", ""))
                                            st.markdown(f"`{seg['start']:.1f}s → {seg['end']:.1f}s`")
                                        with col_b:
                                            score = seg.get("viral_score", 0)
                                            color = "#34d399" if score >= 80 else "#fbbf24" if score >= 60 else "#f87171"
                                            st.markdown(f"<div style='text-align:center;font-size:28px;font-weight:700;color:{color}'>{score}</div><div style='text-align:center;font-size:11px;color:#64748b'>viral score</div>", unsafe_allow_html=True)
                            if important_texts:
                                st.markdown(f"### 📖 Important Text & Key Insights ({len(important_texts)})")
                                for imp in important_texts:
                                    with st.container(border=True):
                                        st.markdown(f"**⏱️ {imp['start']:.1f}s - {imp['end']:.1f}s**")
                                        st.markdown(f"💬 \"{imp.get('important_text', '')}\"")
                                        st.caption(f"📌 {imp.get('reason', '')}")
                                        st.metric("Relevance Score", f"{imp.get('relevance_score', 0)}/100")
                            st.session_state["last_segments"]        = viral_segments
                            st.session_state["last_important_texts"] = important_texts
                    else:
                        segments = data.get("segments", [])
                        if not segments:
                            show_warning("No viral segments were found.")
                        else:
                            st.success(f"✅ {data.get('count')} segments found in {data.get('duration_s')}s")
                            st.session_state["last_segments"] = segments
                            for seg in segments:
                                with st.container(border=True):
                                    col_a, col_b = st.columns([3, 1])
                                    with col_a:
                                        st.markdown(f"**{seg.get('title', '—')}**")
                                        st.caption(seg.get("reason", ""))
                                        st.markdown(f"`{seg['start']:.1f}s → {seg['end']:.1f}s`")
                                    with col_b:
                                        score = seg.get("viral_score", 0)
                                        color = "#34d399" if score >= 80 else "#fbbf24" if score >= 60 else "#f87171"
                                        st.markdown(f"<div style='text-align:center;font-size:28px;font-weight:700;color:{color}'>{score}</div><div style='text-align:center;font-size:11px;color:#64748b'>viral score</div>", unsafe_allow_html=True)

                    with st.expander("Full response JSON"):
                        render_json(data)

                except httpx.HTTPStatusError as exc:
                    st.error(f"HTTP {exc.response.status_code}: {exc.response.text[:400]}")
                except Exception as exc:
                    st.error(str(exc))

    with analyze_tab3:
        st.markdown("**Analyze plain text directly** without timestamps")
        direct_text = st.text_area("Enter text to analyze", height=200,
                                    placeholder="Paste your transcript, article, or any text content here...",
                                    key="direct_text_input")
        extract_important_text = st.checkbox("Extract all important text/quotes", value=True,
                                              key="direct_text_extract_important")
        if direct_text:
            st.caption(f"📝 {len(direct_text)} characters | ~{len(direct_text.split())} words")

        if st.button("🔍 Analyze Text", disabled=not direct_text or len(direct_text.strip()) < 20,
                     key="analyze_text_btn", type="primary", use_container_width=True):
            with st.spinner("Analyzing text content..."):
                try:
                    with httpx.Client(timeout=120) as client:
                        resp = client.post(f"{SERVICES['clip_analyzer']}/analyze-text",
                                           data={"text": direct_text}, headers=HEADERS)
                        resp.raise_for_status()
                        result = resp.json()
                    viral_segments  = result.get('viral_segments', result.get('segments', []))
                    important_texts = result.get('important_texts', [])
                    if not viral_segments and not important_texts:
                        st.warning("No key segments identified.")
                    else:
                        st.success(f"✅ Found {len(viral_segments)} viral segments and {len(important_texts)} important texts!")
                        if viral_segments:
                            st.markdown(f"### 🔥 Viral Segments ({len(viral_segments)})")
                            for idx, seg in enumerate(viral_segments, 1):
                                with st.container(border=True):
                                    col_a, col_b = st.columns([3, 1])
                                    with col_a:
                                        st.markdown(f"**{idx}. {seg.get('title', 'Key Point')}**")
                                        st.caption(seg.get('reason', ''))
                                    with col_b:
                                        score = seg.get('viral_score', 0)
                                        color = "#34d399" if score >= 80 else "#fbbf24" if score >= 60 else "#f87171"
                                        st.markdown(f"<div style='text-align:center;font-size:28px;font-weight:700;color:{color}'>{score}</div><div style='text-align:center;font-size:11px;color:#64748b'>viral score</div>", unsafe_allow_html=True)
                        if important_texts and extract_important_text:
                            st.markdown(f"### 📖 Important Text & Key Quotes ({len(important_texts)})")
                            for idx, imp in enumerate(important_texts, 1):
                                with st.container(border=True):
                                    st.markdown(f"**{idx}. {imp.get('important_text', imp.get('text', ''))[:300]}**")
                                    if imp.get('reason'): st.caption(f"📌 {imp['reason']}")
                                    st.metric("Relevance Score", f"{imp.get('relevance_score', imp.get('viral_score', 0))}/100")
                        st.download_button("💾 Download Analysis (JSON)", json.dumps(result, indent=2),
                                           file_name=f"text_analysis_{int(time.time())}.json",
                                           mime="application/json", key="download_text_analysis")
                except httpx.HTTPStatusError as exc:
                    st.error(f"HTTP Error: {exc.response.status_code}")
                    try: st.code(exc.response.json().get('detail', exc.response.text)[:500])
                    except Exception: st.code(exc.response.text[:500])
                except Exception as exc:
                    st.error(f"Error: {str(exc)}")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — Subtitle Burner
# ─────────────────────────────────────────────────────────────────────────────

with tab_subtitle:
    st.markdown("### Subtitle Burner  —  Add Karaoke Subtitles to Videos")
    st.caption("Upload a video to automatically transcribe and burn karaoke-style subtitles")

    subtitle_tab1, subtitle_tab2 = st.tabs(["🎬 Upload Video (Drag & Drop)", "📁 Use Existing Clips"])

    with subtitle_tab1:
        st.markdown("""
        **Upload a video** to automatically:
        1. Extract audio
        2. Transcribe speech with word-level timestamps
        3. Generate karaoke-style subtitles
        4. Burn subtitles directly into the video
        """)

        uploaded_video = st.file_uploader("Drag & drop or click to upload video",
                                           type=["mp4", "mov", "avi", "mkv", "webm"],
                                           key="subtitle_video_upload")

        st.markdown("### 🎨 Subtitle Styling (Optional)")
        col_style1, col_style2, col_style3 = st.columns(3)
        with col_style1:
            font_size = st.slider("Font Size", min_value=24, max_value=80, value=52, step=2, key="sub_font_size")
        with col_style2:
            font_color = st.selectbox("Font Color",
                                       options=["yellow", "white", "cyan", "magenta", "green", "red", "blue"],
                                       index=0, key="sub_font_color")
        with col_style3:
            words_per_line = st.slider("Words per Line", min_value=2, max_value=8, value=4, step=1, key="sub_words_per_line")

        if uploaded_video:
            file_size_mb = len(uploaded_video.getvalue()) / (1024 ** 2)
            st.info(f"📄 **File:** {uploaded_video.name} | 📦 **Size:** {file_size_mb:.1f} MB")
            with st.expander("🎥 Preview Original Video", expanded=False):
                st.video(uploaded_video)

        if st.button("🔥 Burn Subtitles into Video", disabled=uploaded_video is None,
                     key="burn_direct_btn", type="primary", use_container_width=True):
            with st.spinner("Processing video..."):
                try:
                    progress_bar = st.progress(0)
                    status_text  = st.empty()
                    steps = [(10, "📤 Uploading video..."), (30, "🎵 Analyzing audio..."),
                             (50, "📝 Transcribing with timestamps..."), (70, "🎨 Generating karaoke subtitles..."),
                             (85, "🔥 Burning subtitles into video..."), (95, "✨ Finalizing...")]

                    if font_size != 52 or font_color != "yellow" or words_per_line != 4:
                        endpoint = f"{SERVICES['subtitle_burner']}/burn-with-custom-style"
                        with httpx.Client(timeout=600) as client:
                            files = {"file": (uploaded_video.name, uploaded_video.getvalue(), "video/mp4")}
                            data  = {"font_size": str(font_size), "font_color": font_color,
                                     "words_per_line": str(words_per_line)}
                            for pct, msg in steps:
                                status_text.info(msg); progress_bar.progress(pct); time.sleep(0.2)
                            resp = client.post(endpoint, files=files, data=data, headers=HEADERS)
                            resp.raise_for_status(); result = resp.json()
                    else:
                        endpoint = f"{SERVICES['subtitle_burner']}/burn-video-direct"
                        with httpx.Client(timeout=600) as client:
                            files = {"file": (uploaded_video.name, uploaded_video.getvalue(), "video/mp4")}
                            for pct, msg in steps:
                                status_text.info(msg); progress_bar.progress(pct); time.sleep(0.2)
                            resp = client.post(endpoint, files=files, headers=HEADERS)
                            resp.raise_in_status(); result = resp.json()

                    progress_bar.progress(100); status_text.empty()
                    st.success(f"✅ Subtitles burned successfully in {result.get('duration_s', 0):.1f}s!")

                    col_meta1, col_meta2, col_meta3 = st.columns(3)
                    with col_meta1: st.metric("Job ID", result.get('job_id', 'N/A')[:8])
                    with col_meta2: st.metric("Word Count", result.get('word_count', 0))
                    with col_meta3: st.metric("Status", "Complete")

                    if result.get('transcription'):
                        with st.expander("📝 Transcription Preview", expanded=True):
                            st.text_area("Transcribed Text", result['transcription'], height=150, disabled=True)

                    output_path = result.get('output_video') or result.get('subtitled_video')
                    if output_path and Path(output_path).exists():
                        st.markdown("### 🎬 Video with Subtitles")
                        st.video(output_path)
                        with open(output_path, "rb") as f:
                            st.download_button("💾 Download Subtitled Video", f,
                                               file_name=f"subtitled_{uploaded_video.name}",
                                               mime="video/mp4", key="download_subtitled_video")
                    else:
                        st.warning("Subtitled video file not found at the expected location.")

                    with st.expander("📄 Full Response JSON"):
                        render_json(result)

                except httpx.HTTPStatusError as exc:
                    progress_bar.empty(); st.error(f"HTTP Error {exc.response.status_code}")
                    try: st.code(exc.response.json().get('detail', exc.response.text), language="json")
                    except Exception: st.code(exc.response.text[:500])
                except Exception as exc:
                    progress_bar.empty(); st.error(f"Error: {str(exc)}"); st.exception(exc)

    with subtitle_tab2:
        st.markdown("### Burn subtitles into existing clips from the Clip Extractor")
        use_clips  = st.session_state.get("last_clip_paths") is not None
        clip_source = st.radio("Clip paths source", ["Use clips from Extractor tab", "Enter manually"],
                                disabled=not use_clips, key="sub_clip_source")
        if clip_source == "Use clips from Extractor tab" and use_clips:
            last = st.session_state["last_clip_paths"]
            clip_paths_input = "\n".join(last)
            st.success(f"{len(last)} clip path(s) loaded.")
            st.code(clip_paths_input, language="")
        else:
            clip_paths_input = st.text_area("Clip paths — one per line", height=120,
                                             placeholder="/tmp/reels_shared/clips/abc/clip_01.mp4",
                                             key="sub_paths_manual")

        col_style1b, col_style2b, col_style3b = st.columns(3)
        with col_style1b: font_size_existing = st.slider("Font Size", 24, 80, 52, 2, key="sub_font_size_existing")
        with col_style2b: font_color_existing = st.selectbox("Font Color", ["yellow","white","cyan","magenta","green","red","blue"], index=0, key="sub_font_color_existing")
        with col_style3b: words_per_line_existing = st.slider("Words per Line", 2, 8, 4, 1, key="sub_words_per_line_existing")

        sub_errors = []
        validated_paths = None
        if clip_paths_input.strip():
            validated_paths, paths_err = validate_clip_paths(clip_paths_input)
            if paths_err: sub_errors.append(paths_err)
        if sub_errors: show_errors(sub_errors)

        if st.button("▶ Burn Subtitles into Clips",
                     disabled=(not clip_paths_input.strip() or bool(sub_errors)), key="sub_run"):
            with st.spinner(f"Transcribing & burning {len(validated_paths)} clip(s)…"):
                try:
                    with httpx.Client(timeout=TIMEOUT) as client:
                        resp = client.post(f"{SERVICES['subtitle_burner']}/burn-subtitles",
                                           json={"clip_paths": validated_paths}, headers=HEADERS)
                        resp.raise_for_status(); data = resp.json()
                    subtitled = data.get("subtitled", [])
                    st.success(f"✅ {len(subtitled)} clip(s) subtitled in {data.get('duration_s')}s")
                    for item in subtitled:
                        sub_path = item.get("subtitled_clip", "")
                        exists   = Path(sub_path).exists()
                        st.markdown(f"{'✅' if exists else '⚠️'} `{sub_path}` (words: {item.get('word_count', 0)})")
                        if exists:
                            st.video(sub_path)
                            with open(sub_path, "rb") as f:
                                st.download_button("⬇ Download", f, file_name=Path(sub_path).name,
                                                   mime="video/mp4", key=f"dl_sub_burn_{sub_path}")
                    with st.expander("Full response JSON"): render_json(data)
                except httpx.HTTPStatusError as exc:
                    st.error(f"HTTP {exc.response.status_code}: {exc.response.text[:400]}")
                except Exception as exc:
                    st.error(str(exc))

# ─────────────────────────────────────────────────────────────────────────────
# TAB 6 — Video Resizer
# ─────────────────────────────────────────────────────────────────────────────

with tab_resizer:
    st.markdown("### Video Resizer  —  `POST /resize-to-portrait`")
    st.caption(
        "Upload a **landscape** video to convert it to a portrait format. "
        "Portrait or square videos are rejected with a clear alert."
    )

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        # ── File upload ───────────────────────────────────────────────────
        resizer_upload = st.file_uploader(
            "Upload video  (landscape only)",
            type=ALLOWED_VIDEO_TYPES,
            key="resizer_upload",
            help="MP4, MOV, AVI, MKV — must be landscape (width > height)",
        )

        # ── Live orientation detection (cached, no extra API call) ────────
        resize_orientation = None
        resize_dims        = None
        resizer_file_errors: list[str] = []

        if resizer_upload:
            resizer_file_errors = validate_video_file(resizer_upload)
            if not resizer_file_errors:
                resize_dims        = get_video_dimensions(resizer_upload.getvalue(), resizer_upload.name)
                resize_orientation = get_video_orientation(resizer_upload.getvalue(), resizer_upload.name)

                size_mb = len(resizer_upload.getvalue()) / (1024 ** 2)
                st.caption(f"📄 `{resizer_upload.name}` — {size_mb:.1f} MB")

                if resize_dims:
                    render_orientation_badge(resize_orientation, *resize_dims)

        # ── Preset selector ───────────────────────────────────────────────
        resize_preset = st.selectbox(
            "Target portrait preset",
            options=list(PORTRAIT_PRESET_OPTIONS.keys()),
            format_func=lambda k: PORTRAIT_PRESET_OPTIONS[k],
            key="resizer_preset",
            help="All presets use centre-crop smart scaling. Source audio is preserved.",
        )

        # ── Run button — disabled until a valid landscape file is uploaded ─
        is_landscape_ready = (
            resizer_upload is not None
            and not resizer_file_errors
            and resize_orientation == "landscape"
        )

        resize_run_btn = st.button(
            "🔄 Convert to Portrait",
            disabled=not is_landscape_ready,
            key="resizer_run",
            type="primary",
            use_container_width=True,
        )

    with col_right:
        # ── Inline alerts ─────────────────────────────────────────────────
        if resizer_file_errors:
            show_errors(resizer_file_errors)

        elif resizer_upload and resize_orientation and resize_orientation != "landscape":
            # Non-landscape → show blocking alert
            icon = "📱" if resize_orientation == "portrait" else "⬛"
            st.markdown(
                f"<div class='resize-alert'>"
                f"<div class='resize-alert-title'>{icon} Cannot Convert — Video is Already {resize_orientation.upper()}</div>"
                f"The uploaded video is <strong>{resize_orientation}</strong> "
                f"({resize_dims[0] if resize_dims else '?'} × {resize_dims[1] if resize_dims else '?'} px). "
                f"This module only converts <strong>landscape</strong> videos (width &gt; height).<br><br>"
                f"Please upload a landscape video to use the resizer."
                f"</div>",
                unsafe_allow_html=True,
            )

        elif resizer_upload and resize_orientation == "landscape":
            # Show what will happen
            preset_cfg = {
                "tiktok":    (1080, 1920),
                "instagram": (1080, 1350),
                "square":    (1080, 1080),
            }
            tw, th = preset_cfg.get(resize_preset, (1080, 1920))
            st.markdown("**Conversion preview:**")
            dc1, dc2, dc3 = st.columns([2, 1, 2])
            with dc1:
                st.markdown("<div class='dims-card'><div class='dims-card-label'>INPUT</div>"
                            f"<div class='dims-card-value'>{resize_dims[0] if resize_dims else '?'} × {resize_dims[1] if resize_dims else '?'}</div>"
                            "<div class='dims-card-label' style='margin-top:4px'>landscape</div></div>",
                            unsafe_allow_html=True)
            with dc2:
                st.markdown("<div class='arrow-between'>→</div>", unsafe_allow_html=True)
            with dc3:
                st.markdown(f"<div class='dims-card'><div class='dims-card-label'>OUTPUT</div>"
                            f"<div class='dims-card-value'>{tw} × {th}</div>"
                            f"<div class='dims-card-label' style='margin-top:4px'>{PORTRAIT_PRESET_OPTIONS[resize_preset].split('(')[0].strip()}</div></div>",
                            unsafe_allow_html=True)
            st.caption("ℹ️ Smart centre-crop: scales to fill the target height, then crops the sides.")

        elif not resizer_upload:
            st.info("Upload a landscape video on the left to begin.")

    # ── Conversion result ─────────────────────────────────────────────────────
    if resize_run_btn and is_landscape_ready:
        st.divider()
        prog   = st.progress(0, text="Uploading to resizer service…")
        status = st.empty()

        steps = [
            (20, "📤 Uploading video…"),
            (45, "🔍 Verifying orientation…"),
            (65, "⚙️ Running FFmpeg conversion…"),
            (85, "🎬 Encoding portrait output…"),
            (95, "✅ Finalising…"),
        ]
        for pct, msg in steps:
            status.info(msg)
            prog.progress(pct, text=msg)
            time.sleep(0.25)

        try:
            with httpx.Client(timeout=TIMEOUT) as client:
                resp = client.post(
                    f"{SERVICES['resizer']}/resize-to-portrait",
                    files={"file": (resizer_upload.name, resizer_upload.getvalue(), "video/mp4")},
                    data={"preset": resize_preset},
                    headers=HEADERS,
                )
                resp.raise_for_status()
                data = resp.json()

            prog.progress(100, text="Done!")
            status.empty()

            # ── Success banner ────────────────────────────────────────────
            st.markdown(
                f"<div class='resize-success-box'>"
                f"<strong>✅ Conversion complete</strong> in {data.get('duration_s', 0):.1f}s &nbsp;·&nbsp; "
                f"Job <code>{data.get('job_id')}</code> &nbsp;·&nbsp; "
                f"Preset <code>{data.get('preset_label', resize_preset)}</code>"
                f"</div>",
                unsafe_allow_html=True,
            )

            # ── Metrics ───────────────────────────────────────────────────
            m1, m2, m3, m4 = st.columns(4)
            inp = data.get("input_dims",  {})
            out = data.get("output_dims", {})
            m1.metric("Input",        f"{inp.get('width','?')} × {inp.get('height','?')}")
            m2.metric("Output",       f"{out.get('width','?')} × {out.get('height','?')}")
            m3.metric("Output size",  f"{data.get('output_size_mb', 0):.1f} MB")
            m4.metric("Time",         f"{data.get('duration_s', 0):.1f}s")

            # ── Video player + download ───────────────────────────────────
            output_path = data.get("output_path", "")
            if output_path and Path(output_path).exists():
                st.markdown("### 🎬 Converted Portrait Video")
                st.video(output_path)
                with open(output_path, "rb") as vf:
                    st.download_button(
                        "⬇ Download Portrait Video",
                        vf,
                        file_name=Path(output_path).name,
                        mime="video/mp4",
                        key="resizer_download",
                    )
            else:
                st.warning(
                    f"Conversion succeeded on the server but the output file was not found at:\n`{output_path}`\n\n"
                    "Check that shared storage is mounted correctly."
                )

            with st.expander("Full response JSON"):
                render_json(data)

        except httpx.HTTPStatusError as exc:
            prog.empty()
            try:
                detail = exc.response.json().get("detail", exc.response.text)
            except Exception:
                detail = exc.response.text
            # 400 means wrong orientation — show the styled alert, not a generic error
            if exc.response.status_code == 400:
                st.markdown(
                    f"<div class='resize-alert'>"
                    f"<div class='resize-alert-title'>⛔ Server Rejected the Video</div>"
                    f"{detail}"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.error(f"HTTP {exc.response.status_code}: {detail}")
        except Exception as exc:
            prog.empty()
            st.error(f"Error: {exc}")

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────

st.divider()
st.markdown(
    "<div style='text-align:center;color:#334155;font-size:12px;"
    "font-family:JetBrains Mono,monospace'>"
    "orchestrator :8000 · audio :8001 · transcriber :8002 · "
    "analyzer :8003 · extractor :8004 · subtitler :8005 · resizer :8006"
    "</div>",
    unsafe_allow_html=True,
)