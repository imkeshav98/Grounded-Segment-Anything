import asyncio
from fastapi import HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

class TimeoutMiddleware(BaseHTTPMiddleware):
    """Middleware to handle request timeouts"""
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """Process the request with a timeout"""
        try:
            return await asyncio.wait_for(call_next(request), timeout=300)
        except asyncio.TimeoutError:
            return Response("Request timeout", status_code=504)