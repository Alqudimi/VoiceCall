"""API authentication (bearer token) + request-scoped security helpers."""
from __future__ import annotations

import secrets
from pathlib import Path

from fastapi import HTTPException, Request, WebSocket
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from rvc_platform.core.errors import SecurityError

_bearer = HTTPBearer(auto_error=False)


class TokenAuth:
    """Bearer-token guard.

    Disabled by default on localhost deployments; automatically *required*
    when the service is exposed through a Cloudflare tunnel (see
    ``tunnel_service`` — it enables auth unless a token was configured).
    """

    def __init__(self, token: str, enabled: bool) -> None:
        self._token = token
        self.enabled = enabled

    def enable(self, token: str) -> None:
        """Enable authentication only after a non-empty secret is available."""
        if not token:
            raise SecurityError("refusing to enable authentication with an empty token")
        self._token = token
        self.enabled = True

    def check_header(self, header: str | None) -> bool:
        if not self.enabled:
            return True
        if not header or not header.startswith("Bearer "):
            return False
        candidate = header[7:].strip()
        return secrets.compare_digest(candidate, self._token)

    def guard_http(self, request: Request) -> None:
        if not self.enabled:
            return
        header = request.headers.get("Authorization")
        if not self.check_header(header):
            raise HTTPException(status_code=401, detail="missing or invalid bearer token")

    async def guard_ws(self, ws: WebSocket) -> bool:
        if not self.enabled:
            return True
        header = ws.headers.get("Authorization")
        if self.check_header(header):
            return True
        await ws.close(code=4401, reason="unauthorized")
        return False


def issue_token(length: int = 24) -> str:
    return secrets.token_urlsafe(length)
