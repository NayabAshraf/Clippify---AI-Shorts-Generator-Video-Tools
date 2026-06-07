"""
shared/http_client.py — Reusable async HTTP client and auth utilities.

Usage (inside any microservice):
    from shared.http_client import ServiceClient, require_internal_token

    # Calling another service:
    async with ServiceClient("audio_extractor") as client:
        resp = await client.post("/extract-audio", files={"file": ...})

    # Protecting an endpoint:
    @app.post("/my-endpoint")
    async def handler(request: Request, _=Depends(require_internal_token)):
        ...
"""

import httpx
from fastapi import Depends, Header, HTTPException, Request
from shared.config import INTERNAL_TOKEN, SERVICE_URLS

# ── Timeout tuning ────────────────────────────────────────────────────────────
# Audio extraction and transcription can take minutes on long videos.
DEFAULT_TIMEOUT = httpx.Timeout(
    connect=10.0,
    read=1800.0,   # 30 min read — covers transcription of long videos
    write=300.0,  # 5 min write — covers uploading large video files
    pool=5.0,
)


class ServiceClient:
    """
    Async HTTP client pre-configured for inter-service calls.

    Automatically injects the shared internal auth token header so you
    never need to remember it at the call site.

        async with ServiceClient("clip_analyzer") as client:
            data = await client.post_json("/analyze", json=payload)
    """

    def __init__(self, service_name: str, timeout: httpx.Timeout = DEFAULT_TIMEOUT):
        base_url = SERVICE_URLS.get(service_name)
        if not base_url:
            raise ValueError(
                f"Unknown service '{service_name}'. "
                f"Valid names: {list(SERVICE_URLS.keys())}"
            )
        self._base_url = base_url
        self._timeout  = timeout
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "ServiceClient":
        headers = {}
        if INTERNAL_TOKEN:
            headers["X-Internal-Token"] = INTERNAL_TOKEN
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers=headers,
            timeout=self._timeout,
        )
        return self

    async def __aexit__(self, *_):
        if self._client:
            await self._client.aclose()

    async def post(self, path: str, **kwargs) -> httpx.Response:
        resp = await self._client.post(path, **kwargs)
        resp.raise_for_status()
        return resp

    async def post_json(self, path: str, json: dict, **kwargs) -> dict:
        resp = await self.post(path, json=json, **kwargs)
        return resp.json()

    async def get(self, path: str, **kwargs) -> httpx.Response:
        resp = await self._client.get(path, **kwargs)
        resp.raise_for_status()
        return resp

    async def health(self) -> bool:
        """Returns True if the remote service is healthy."""
        try:
            resp = await self.get("/health")
            return resp.json().get("status") == "ok"
        except Exception:
            return False


# ── Auth dependency ───────────────────────────────────────────────────────────

async def require_internal_token(
    x_internal_token: str = Header(default=""),
) -> None:
    """
    FastAPI dependency that enforces the shared internal auth token.

    Add as a dependency on any endpoint that should only be called
    by other services (not the public internet).

    Skip verification if INTERNAL_TOKEN is empty (dev mode).
    """
    if not INTERNAL_TOKEN:
        return   # auth disabled in dev
    if x_internal_token != INTERNAL_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid internal token")