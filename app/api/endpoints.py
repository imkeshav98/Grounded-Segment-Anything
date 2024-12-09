# app/api/endpoints.py
import logging
from fastapi import APIRouter, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse

from ..config import config
from .models import ProcessingResponse, ProcessingStatus
from ..core.utils import validate_file_size, validate_file_type, async_timeout, managed_resource
from .. import processor  # Import from main app package

router = APIRouter()

@router.post("/process_image", response_model=ProcessingResponse)
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
        
        async with managed_resource():
            if not processor:
                raise HTTPException(status_code=503, detail="Image processor not initialized")
                
            result = processor.process_image(content, prompt, auto_detect_text)
            
            if result.status == ProcessingStatus.ERROR:
                raise HTTPException(status_code=500, detail=result.message)
                
            return result
            
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error processing upload: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during image processing")
    finally:
        await file.close()

@router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": "2.0.0",
        "device": str(processor.device if processor else "not initialized")
    }

@router.post("/cleanup")
async def force_cleanup():
    try:
        if processor:
            processor._cleanup_resources()
        return {"status": "success", "message": "Cleanup completed successfully"}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Cleanup failed: {str(e)}"
        )