# File: app/main.py

from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from app.config import config
from app.core.processor import ImageProcessor
from app.models.schemas import ProcessingResponse, ProcessingStatus
from app.utils.middleware import TimeoutMiddleware, async_timeout

# Global processor instance
processor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events"""
    global processor
    try:
        processor = ImageProcessor(config)
        yield
    except Exception as e:
        raise
    finally:
        if processor:
            processor.cleanup_resources()

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

def validate_file_size(content_length: int):
    if content_length > config.MAX_CONTENT_LENGTH:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {config.MAX_CONTENT_LENGTH/1024/1024}MB"
        )

def validate_file_type(filename: str):
    file_extension = filename.split(".")[-1].lower() if filename else ""
    if not file_extension or file_extension not in config.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed types are: {', '.join(config.ALLOWED_EXTENSIONS)}"
        )

@app.post("/api/v2/process_image", response_model=ProcessingResponse)
@async_timeout(300)
async def process_image(
    file: UploadFile = File(...),
    prompt: str = Form(...),
    auto_detect_text: bool = Form(False)
) -> ProcessingResponse:
    try:
        validate_file_type(file.filename)
        content = await file.read()
        validate_file_size(len(content))
        
        result = processor.process_image(content, prompt, auto_detect_text)
        
        if result.status == ProcessingStatus.ERROR:
            raise HTTPException(status_code=500, detail=result.message)
            
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error during image processing")
    finally:
        await file.close()

@app.get("/api/v2/health")
async def health_check():
    return {
        "status": "healthy",
        "version": app.version,
        "device": str(processor.device if processor else "not initialized")
    }

@app.post("/api/v2/cleanup")
async def force_cleanup():
    try:
        if processor:
            processor.cleanup_resources()
        return {"status": "success", "message": "Cleanup completed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "message": exc.detail,
            "code": exc.status_code
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": "Internal server error",
            "code": 500
        }
    )

if __name__ == "__main__":
    import uvicorn
    
    uvicorn_config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        timeout_keep_alive=30,
        limit_concurrency=10,
        timeout_notify=30,
        timeout_graceful_shutdown=30
    )
    
    server = uvicorn.Server(uvicorn_config)
    server.run()