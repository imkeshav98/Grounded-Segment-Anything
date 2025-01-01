# File: app/main.py

import sys
import os
import base64
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv

from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any

from app.config import config
from app.core.processor import ImageProcessor
from app.core.model_manager import model_manager
from app.models.schemas import ProcessingResponse, ProcessingStatus, DetectedObject, ThemeProperties, LayerType
from app.utils.middleware import TimeoutMiddleware
from app.core.vision_processor import VisionProcessor
from app.core.inpaint_processor import InpaintAPIClient

# Load environment variables first
load_dotenv()

# Create a class to hold global instances
class GlobalInstances:
    processor = None
    model_manager = None

# Initialize global instances container
globals_container = GlobalInstances()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events"""
    try:
        print("Initializing model manager...")
        globals_container.model_manager = model_manager
        
        print("Initializing processor...")
        globals_container.processor = ImageProcessor(config)
        
        print("Startup complete")
        yield
    except Exception as e:
        print(f"Error during startup: {str(e)}")
        raise
    finally:
        try:
            if globals_container.processor:
                globals_container.processor.cleanup_resources()
            if globals_container.model_manager:
                globals_container.model_manager.cleanup()
        except Exception as cleanup_error:
            print(f"Error during cleanup: {cleanup_error}")

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

def ensure_initialized():
    """Ensure the processor is initialized before processing requests"""
    if not globals_container.processor:
        raise HTTPException(
            status_code=503,
            detail="Service is initializing. Please try again in a moment."
        )

@app.post("/api/v2/analyze_image")
async def analyze_image(
    file: UploadFile = File(...)
) -> Dict[str, Any]:
    """Analyze image with OpenAI Vision to generate initial prompt"""
    try:
        ensure_initialized()
        content = await file.read()
        validate_file_type(file.filename)
        validate_file_size(len(content))
        
        # Use Vision to analyze and return prompt
        vision_processor = VisionProcessor()
        result = await vision_processor.analyze_image(content)
        
        return result
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error analyzing image: {str(e)}"
        )

@app.post("/api/v2/process_image", response_model=ProcessingResponse)
async def process_image(
    file: UploadFile = File(...),
    prompt: str = Form(...),
    auto_detect_text: bool = Form(True)
) -> ProcessingResponse:
    try:
        ensure_initialized()
        
        # --- Input Validation ---
        content = await file.read()
        validate_file_type(file.filename)
        validate_file_size(len(content))

        # --- Step 1: Initial Processing ---
        result = globals_container.processor.process_image(
            image_content=content,
            prompt=prompt,
            auto_detect_text=auto_detect_text
        )
        print("Initial processing completed")

        # Exit early if no objects detected or processing failed
        if not (result.status == ProcessingStatus.SUCCESS and result.objects):
            return result

        # --- Step 2: Object Validation ---
        vision_processor = VisionProcessor()
        visualization_image = base64.b64decode(result.visualization)
        
        validated_objects = await vision_processor.validate_detections(
            visualization_image,
            [obj.model_dump() for obj in result.objects]
        )
        print("Validation completed")

        # Handle case where no objects pass validation
        if not validated_objects:
            return ProcessingResponse(
                status=ProcessingStatus.ERROR,
                message="No valid detections after validation"
            )

        # --- Step 3: Object Classification ---
        image_objects = [
            obj for obj in validated_objects 
            if obj["layer_type"] == LayerType.IMAGE
        ]
        other_objects = [
            obj for obj in validated_objects 
            if obj["layer_type"] != LayerType.IMAGE
        ]

        # --- Step 4: Style Enhancement ---
        enhanced_data = await vision_processor.enhance_styles(
            visualization_image,
            other_objects
        )

        # --- Step 5: Result Assembly ---
        result.objects = [
            DetectedObject(**obj) 
            for obj in enhanced_data["elements"]
        ]
        result.objects.extend([
            DetectedObject(**obj) 
            for obj in image_objects
        ])

        # --- Step 6: Output Generation ---
        result = globals_container.processor.regenerate_outputs(
            image_content=content,
            validated_objects=result.objects
        )
        print("Regeneration completed")

        # --- Step 7: Inpainting ---
        inpaint_client = InpaintAPIClient()
        result.inpainted_image = await inpaint_client.inpaint_image(
            original_image_url=result.original_image,
            mask_image_url=result.masked_output
        )
        print("Inpainting completed")

        # --- Step 8: Usage Statistics ---
        result.usage = vision_processor.get_total_usage()

        return result

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@app.get("/api/v2/health")
async def health_check():
    return {
        "status": "healthy",
        "version": app.version,
        "device": str(globals_container.processor.device if globals_container.processor else "not initialized"),
        "model_manager_status": "initialized" if globals_container.model_manager else "not initialized",
        "processor_status": "initialized" if globals_container.processor else "not initialized"
    }

@app.post("/api/v2/cleanup")
async def force_cleanup():
    try:
        if globals_container.processor:
            globals_container.processor.cleanup_resources()
        if globals_container.model_manager:
            globals_container.model_manager.cleanup()
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