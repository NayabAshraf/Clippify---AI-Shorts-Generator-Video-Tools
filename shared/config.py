"""
shared/config.py
Loads .env automatically — no need to set env vars in the shell.
"""
import os
import sys
from pathlib import Path

# ── Load .env from project root ───────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    _env_file = Path(__file__).parent.parent / ".env"
    load_dotenv(_env_file)
    print(f"[config] Loaded .env from {_env_file}")
except ImportError:
    print("[config] python-dotenv not installed — using shell env vars only")

# ── Ports ─────────────────────────────────────────────────────────────────────
AUDIO_EXTRACTOR_PORT = int(os.getenv("AUDIO_EXTRACTOR_PORT", "8001"))
TRANSCRIBER_PORT     = int(os.getenv("TRANSCRIBER_PORT",     "8002"))
CLIP_ANALYZER_PORT   = int(os.getenv("CLIP_ANALYZER_PORT",   "8003"))
CLIP_EXTRACTOR_PORT  = int(os.getenv("CLIP_EXTRACTOR_PORT",  "8004"))
SUBTITLE_BURNER_PORT = int(os.getenv("SUBTITLE_BURNER_PORT", "8005"))
ORCHESTRATOR_PORT    = int(os.getenv("ORCHESTRATOR_PORT",    "8000"))
TTS_PORT             = int(os.getenv("TTS_PORT",             "8006"))  # Added for TTS

_HOST = os.getenv("SERVICE_HOST", "http://localhost")

# ── Service URLs ──────────────────────────────────────────────────────────────
SERVICE_URLS: dict[str, str] = {
    "audio_extractor": os.getenv("AUDIO_EXTRACTOR_URL", f"{_HOST}:{AUDIO_EXTRACTOR_PORT}"),
    "transcriber":     os.getenv("TRANSCRIBER_URL",     f"{_HOST}:{TRANSCRIBER_PORT}"),
    "clip_analyzer":   os.getenv("CLIP_ANALYZER_URL",   f"{_HOST}:{CLIP_ANALYZER_PORT}"),
    "clip_extractor":  os.getenv("CLIP_EXTRACTOR_URL",  f"{_HOST}:{CLIP_EXTRACTOR_PORT}"),
    "subtitle_burner": os.getenv("SUBTITLE_BURNER_URL", f"{_HOST}:{SUBTITLE_BURNER_PORT}"),
    "tts":             os.getenv("TTS_URL",             f"{_HOST}:{TTS_PORT}"),  # Added for TTS
}

# ── Shared storage ────────────────────────────────────────────────────────────
# Windows-safe default — avoids creating a literal \tmp folder at drive root
_default_storage = r"C:\reels_shared" if sys.platform == "win32" else "/tmp/reels_shared"
SHARED_STORAGE_ROOT = os.getenv("SHARED_STORAGE_ROOT", _default_storage)

# Create shared storage if it doesn't exist
Path(SHARED_STORAGE_ROOT).mkdir(parents=True, exist_ok=True)

# ── FFmpeg ────────────────────────────────────────────────────────────────────
FFMPEG_BIN  = os.getenv("FFMPEG_BIN",  r"C:\ffmpeg\bin\ffmpeg.exe")
FFPROBE_BIN = os.getenv("FFPROBE_BIN", r"C:\ffmpeg\bin\ffprobe.exe")

# Add FFmpeg to PATH if it exists
def setup_ffmpeg_path():
    """Add FFmpeg directory to system PATH"""
    ffmpeg_dir = os.path.dirname(FFMPEG_BIN)
    if os.path.exists(ffmpeg_dir) and ffmpeg_dir not in os.environ.get("PATH", ""):
        os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
        print(f"[config] Added FFmpeg to PATH: {ffmpeg_dir}")
    elif not os.path.exists(ffmpeg_dir):
        print(f"[config] Warning: FFmpeg not found at {ffmpeg_dir}")
        print("[config] Please install FFmpeg or update FFMPEG_BIN in .env")

# Call this function
setup_ffmpeg_path()

# ── Inter-service auth ────────────────────────────────────────────────────────
INTERNAL_TOKEN = os.getenv("INTERNAL_TOKEN", "dev-secret-change-me")

# ── Groq ──────────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# ── TTS Configuration ─────────────────────────────────────────────────────────
# Redis settings
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
REDIS_URL = os.getenv("REDIS_URL", f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}")

# Celery settings
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", REDIS_URL)
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", REDIS_URL)

# Audio settings
AUDIO_TTL_SECONDS = int(os.getenv("AUDIO_TTL_SECONDS", "600"))  # 10 minutes
MAX_AUDIO_SIZE_MB = int(os.getenv("MAX_AUDIO_SIZE_MB", "25"))
MAX_TEXT_LENGTH = int(os.getenv("MAX_TEXT_LENGTH", "5000"))

# TTS Model settings (if using local TTS)
TTS_MODEL_PATH = os.getenv("TTS_MODEL_PATH", "")
USE_GPU_TTS = os.getenv("USE_GPU_TTS", "false").lower() == "true"

# ── Helper Functions ──────────────────────────────────────────────────────────
def get_ffmpeg_path() -> str:
    """Return FFmpeg executable path"""
    if os.path.exists(FFMPEG_BIN):
        return FFMPEG_BIN
    # Try to find in PATH
    import shutil
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        return ffmpeg_path
    raise FileNotFoundError(f"FFmpeg not found at {FFMPEG_BIN} or in PATH")

def get_ffprobe_path() -> str:
    """Return FFprobe executable path"""
    if os.path.exists(FFPROBE_BIN):
        return FFPROBE_BIN
    # Try to find in PATH
    import shutil
    ffprobe_path = shutil.which("ffprobe")
    if ffprobe_path:
        return ffprobe_path
    raise FileNotFoundError(f"FFprobe not found at {FFPROBE_BIN} or in PATH")

def validate_ffmpeg():
    """Validate FFmpeg installation"""
    try:
        import subprocess
        ffmpeg_path = get_ffmpeg_path()
        result = subprocess.run(
            [ffmpeg_path, "-version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            version_line = result.stdout.split('\n')[0]
            print(f"[config] ✅ FFmpeg found: {version_line}")
            return True
        else:
            print(f"[config] ❌ FFmpeg error: {result.stderr}")
            return False
    except Exception as e:
        print(f"[config] ❌ FFmpeg validation failed: {e}")
        return False

# Run FFmpeg validation on import
if __name__ != "__main__":
    validate_ffmpeg()