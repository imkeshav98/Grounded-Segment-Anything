# app/main.py
import uvicorn
import signal
import sys
import logging
import torch
import matplotlib.pyplot as plt
import gc
from app import create_app, processor  # Import from app package directly

app = create_app()

def handle_shutdown(signum, frame):
    """Handle graceful shutdown"""
    logging.info("Received shutdown signal, cleaning up...")
    if processor:
        processor._cleanup_resources()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    plt.close('all')
    sys.exit(0)

# Register shutdown handlers
signal.signal(signal.SIGINT, handle_shutdown)
signal.signal(signal.SIGTERM, handle_shutdown)

if __name__ == "__main__":
    # Configure uvicorn with improved settings
    uvicorn_config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        timeout_keep_alive=30,
        limit_concurrency=10,
        timeout_notify=30,
        timeout_graceful_shutdown=30,
        log_config={
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "()": "uvicorn.logging.DefaultFormatter",
                    "fmt": "%(levelprefix)s %(asctime)s %(message)s",
                    "datefmt": "%Y-%m-%d %H:%M:%S",
                },
            },
            "handlers": {
                "default": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stderr",
                },
            },
            "loggers": {
                "uvicorn": {"handlers": ["default"], "level": "INFO"},
            },
        },
    )
    
    # Start server
    server = uvicorn.Server(uvicorn_config)
    server.run()