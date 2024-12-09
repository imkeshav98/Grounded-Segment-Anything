# app/__init__.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .config import AppConfig, config  # Make sure we import both the class and instance
from .middleware.timeout import TimeoutMiddleware
from .core.processor import ImageProcessor
from contextlib import asynccontextmanager
import logging

processor = None  # Initialize global processor variable

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events"""
    global processor
    
    # Startup
    try:
        processor = ImageProcessor(config)
        logging.info("Image processor initialized successfully")
        yield
    except Exception as e:
        logging.error(f"Failed to initialize image processor: {str(e)}")
        raise RuntimeError("Failed to initialize image processor") from e
    finally:
        # Shutdown
        if processor:
            processor._cleanup_resources()
        logging.info("Cleanup completed")

def create_app():
    """Create and configure the FastAPI application"""
    app = FastAPI(
        title="Image Processing API",
        version="2.0.0",
        description="API for image processing with object detection and text recognition",
        lifespan=lifespan
    )
    
    # Add middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(TimeoutMiddleware)
    
    # Include API routes
    from .api.endpoints import router
    app.include_router(router, prefix="/api/v2")
    
    return app

# Make necessary items available for import
__all__ = ['create_app', 'config', 'processor']