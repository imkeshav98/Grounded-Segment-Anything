import logging
from fastapi import APIRouter, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from typing import Dict, Any

from ..core.processor import processor
from ..config import config
from .models import ProcessingResponse, ProcessingStatus
from ..core.utils import (
    validate_file_size,
    validate_file_type,
    async_timeout,
    managed_resource
)

router = APIRouter()

@router.post("/process_image", response_model=ProcessingResponse)
@async_timeout(300)  # 5 minutes timeout
async def process_image(
    file: UploadFile = File(...),
    prompt: str = Form(...),
    auto_detect_text: bool = Form(False)
) -> ProcessingResponse:
    """
    Process an image with object detection and optional text recognition.
    
    Args:
        file: Uploaded image file
        prompt: Text prompt for object detection
        auto_detect_text: Whether to perform OCR on detected regions
        
    Returns:
        ProcessingResponse object containing detection results
    """
    try:
        # Validate request
        validate_file_type(file.filename)
        content = await file.read()
        validate_file_size(len(content))
        
        # Process image with resource management
        async with managed_resource():
            result = processor.process_image(content, prompt, auto_detect_text)
            
            if result.status == ProcessingStatus.ERROR:
                raise HTTPException(status_code=500, detail=result.message)
                
            return result
            
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error processing upload: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Internal server error during image processing"
        )
    finally:
        await file.close()

@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Check API health status.
    
    Returns:
        Dictionary containing health status information
    """
    return {
        "status": "healthy",
        "version": "2.0.0",
        "device": str(processor.device if processor else "not initialized")
    }

@router.post("/cleanup")
async def force_cleanup() -> Dict[str, str]:
    """
    Force cleanup of system resources.
    
    Returns:
        Dictionary containing cleanup status
    """
    try:
        if processor:
            processor._cleanup_resources()
        return {
            "status": "success",
            "message": "Cleanup completed successfully"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Cleanup failed: {str(e)}"
        )

@router.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "message": exc.detail,
            "code": exc.status_code
        }
    )

@router.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    logging.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": "Internal server error",
            "code": 500
        }
    )