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
from app.models.schemas import ProcessingResponse, ProcessingStatus, DetectedObject, ThemeProperties
from app.utils.middleware import TimeoutMiddleware
from app.core.vision_processor import VisionProcessor

# Global processor instance
processor = None
load_dotenv()

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
    
@app.post("/api/v2/analyze_image")
async def analyze_image(
    file: UploadFile = File(...)
) -> Dict[str, Any]:
    """Analyze image with OpenAI Vision to generate initial prompt"""
    try:
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
        content = await file.read()
        validate_file_type(file.filename)
        validate_file_size(len(content))
        
        print("\n=== Step 1: Initial Processing ===")
        # Step 1: Initial processing
        result = processor.process_image(content, prompt, auto_detect_text)
        print("Initial Objects:")
        for obj in result.objects:
            print(f"ID: {obj.object_id}, Type: {obj.object}, BBox: {obj.bbox}")
        
        if result.status == ProcessingStatus.SUCCESS and result.objects:
            print("\n=== Step 2: Vision Validation ===")
            # Step 2: Validate using visualization
            visualization_image = base64.b64decode(result.visualization)
            vision_processor = VisionProcessor()
            validated_objects = await vision_processor.validate_detections(
                visualization_image,
                [obj.dict() for obj in result.objects]
            )
            
            print("Validated Objects:")
            for obj in validated_objects:
                print(f"ID: {obj['object_id']}, Type: {obj['object']}, BBox: {obj['bbox']}")
            
            if validated_objects:
                print("\n=== Step 3: Style Enhancement ===")
                # Step 3: Enhance with styles
                enhanced_data = await vision_processor.enhance_styles(
                    visualization_image,
                    validated_objects
                )
                
                print("Enhanced Objects:")
                for elem in enhanced_data["elements"]:
                    print(f"ID: {elem['object_id']}, Type: {elem['object']}, BBox: {elem['bbox']}, Styles: {elem.get('styles', None)}")
                
                # Update result
                result.objects = [DetectedObject(**obj) for obj in enhanced_data["elements"]]
                if "theme" in enhanced_data:
                    result.theme = ThemeProperties(**enhanced_data["theme"])
                    print("\nTheme:", enhanced_data["theme"])
                
                print("\n=== Step 4: Regenerating Outputs ===")
                # Regenerate outputs
                result = processor.regenerate_outputs(content, result.objects)
                print("Final Objects:")
                for obj in result.objects:
                    print(f"ID: {obj.object_id}, Type: {obj.object}, BBox: {obj.bbox}")
            else:
                result.status = ProcessingStatus.ERROR
                result.message = "No valid detections after validation"
                print("\nError: No valid detections after validation")
        
        return result
            
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
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