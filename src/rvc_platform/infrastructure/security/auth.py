"""API authentication (bearer token) + request-scoped security helpers."""
from __future__ import annotations

import secrets

from fastapi import HTTPException, Request, WebSocket
from fastapi.security import HTTPBearer


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
