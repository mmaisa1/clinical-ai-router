# app/auth.py
"""
Admin authentication via X-Admin-Key header.

Used as a FastAPI dependency on all /admin/* routes. 
Public routes (/classify, /classify/batch, /health) do not use this.
"""

from fastapi import Header, HTTPException, status

from app.config import ADMIN_API_KEY


async def require_admin_key(x_admin_key: str = Header(...)) -> None:
    """
    FastAPI dependency that validates the X-Admin-Key header.

    Usage:
        @router.get("/admin/metrics", dependencies=[Depends(require_admin_key)])
    """
    if x_admin_key != ADMIN_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-Admin-Key header.",
        )