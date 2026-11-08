"""
CallTrust Engine – FastAPI Application Entry Point

Configures CORS, mounts REST + WebSocket routers, and provides
Prometheus metrics middleware.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api import api_router, ws_router
from .config import settings

from contextlib import asynccontextmanager

# ─── Logging ──────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s │ %(name)-25s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("calltrust")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"🛡️  {settings.app_name} starting")
    logger.info(f"   Mock inference: {settings.mock_inference}")
    logger.info(f"   Privacy mode: {settings.privacy_mode}")
    logger.info(f"   Update interval: {settings.update_interval_ms}ms")
    yield
    logger.info(f"🛡️  {settings.app_name} shutting down")


# ─── App Factory ──────────────────────────────────────
def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        description="Real-time deepfake voice detection for live calls",
        version="0.1.0",
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Prometheus Metrics endpoint
    if settings.prometheus_enabled:
        from prometheus_client import make_asgi_app
        metrics_app = make_asgi_app()
        app.mount("/metrics", metrics_app)

    # Routers
    app.include_router(ws_router)
    app.include_router(api_router)

    return app


app = create_app()
