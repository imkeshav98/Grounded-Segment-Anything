# File: app/main.py

import sys
import os
import base64
import logging
import logging.handlers
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any, List
from pydantic import BaseModel
import aiohttp

from app.config import config
from app.core.processor import ImageProcessor
from app.core.model_manager import model_manager
from app.models.schemas import ProcessingResponse, ProcessingStatus, DetectedObject, ThemeProperties, LayerType
from app.utils.middleware import TimeoutMiddleware
from app.core.vision_processor import VisionProcessor
from app.core.inpaint_processor import InpaintAPIClient
from app.core.prompt_processor import PromptProcessor
from app.core.replicate_client import ReplicateClient

# Load environment variables first
load_dotenv()

# Configure logging
def setup_logging():
    """Configure logging with rotation and proper formatting"""
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Create formatters
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    file_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
    )

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # Rotating file handler
    file_handler = logging.handlers.RotatingFileHandler(
        os.path.join(log_dir, 'app.log'),
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    # Create separate error log
    error_handler = logging.handlers.RotatingFileHandler(
        os.path.join(log_dir, 'error.log'),
        maxBytes=10*1024*1024,
        backupCount=5
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(file_formatter)
    root_logger.addHandler(error_handler)

    return root_logger

logger = setup_logging()

class GenerationRequest(BaseModel):
    prompt: str
    tone: str
    style: str
    brandName: str
    selectedColors: List[str]
    autoDetectText: bool = True
    orgImage: str = None

class GlobalInstances:
    processor = None
    model_manager = None

globals_container = GlobalInstances()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events"""
    logger.info("Starting application initialization...")
    try:
        globals_container.model_manager = model_manager
        logger.info("Model manager initialized successfully")
        
        globals_container.processor = ImageProcessor(config)
        logger.info("Image processor initialized successfully")
        
        logger.info("Application startup completed successfully")
        yield
    except Exception as e:
        logger.error(f"Critical error during startup: {str(e)}", exc_info=True)
        raise
    finally:
        logger.info("Starting cleanup process...")
        try:
            if globals_container.processor:
                globals_container.processor.cleanup_resources()
                logger.info("Processor resources cleaned up successfully")
            if globals_container.model_manager:
                globals_container.model_manager.cleanup()
                logger.info("Model manager cleaned up successfully")
        except Exception as cleanup_error:
            logger.error(f"Error during cleanup: {cleanup_error}", exc_info=True)

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
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
)

app.add_middleware(TimeoutMiddleware)

def ensure_initialized():
    """Ensure the processor is initialized before processing requests"""
    if not globals_container.processor:
        logger.error("Service accessed before initialization completed")
        raise HTTPException(
            status_code=503,
            detail="Service is initializing. Please try again in a moment."
        )

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Middleware to log all incoming requests and their processing time"""
    start_time = datetime.now()
    
    # Generate request ID
    request_id = os.urandom(6).hex()
    logger.info(f"Request {request_id} started - Method: {request.method} Path: {request.url.path}")
    
    response = await call_next(request)
    
    # Calculate processing time
    process_time = (datetime.now() - start_time).total_seconds()
    logger.info(f"Request {request_id} completed - Status: {response.status_code} Processing Time: {process_time:.3f}s")
    
    return response

@app.post("/api/v2/process_image", response_model=ProcessingResponse)
async def process_image(
    request: GenerationRequest
) -> ProcessingResponse:
    logger.info(f"Processing new image request - Prompt: {request.prompt[:50]}...")
    try:
        ensure_initialized()

        isImageProvided = request.orgImage is not None

        if isImageProvided:
            # --- Step 0: Download Image ---
            logger.debug("Downloading image")
            async with aiohttp.ClientSession() as session:
                async with session.get(request.orgImage) as response:
                    image_content = await response.read()
            logger.info("Image downloaded successfully")
        else:
            # --- Step 1: Generate Optimized Prompt ---
            logger.debug("Starting prompt generation")
            prompt_processor = PromptProcessor()
            prompt_result = await prompt_processor.generate_prompt(
                base_prompt=request.prompt,
                tone=request.tone,
                style=request.style,
                brand_name=request.brandName,
                selected_colors=request.selectedColors
            )
            logger.info("Prompt generated successfully")

            # --- Step 2: Generate Image ---
            logger.debug("Starting image generation")
            replicate_client = ReplicateClient()
            image_content = replicate_client.generate_image(prompt_result["prompt"])[0]
            logger.info("Image generated successfully")

        if not image_content:
            logger.error("Image content is empty")
            return ProcessingResponse(
                status=ProcessingStatus.ERROR,
                message="Image content is empty"
            )
        

        # --- Step 3: Analyze Image ---
        logger.debug("Starting image analysis")
        vision_processor = VisionProcessor()
        analysis_result = await vision_processor.analyze_image(image_content, prompt_result["prompt"])

        if not analysis_result["prompt"]:
            logger.error("Image analysis failed - no prompt generated")
            return ProcessingResponse(
                status=ProcessingStatus.ERROR,
                message="Image analysis failed"
            )
        logger.info("Image analysis completed successfully")

        # --- Step 4: Initial Processing ---
        logger.debug("Starting initial image processing")
        result = globals_container.processor.process_image(
            image_content=image_content,
            prompt=analysis_result["prompt"],
            auto_detect_text=request.autoDetectText
        )

        logger.info(f"Initial processing completed - Status: {result.status}")

        if not (result.status == ProcessingStatus.SUCCESS and result.objects):
            logger.warning("No objects detected or processing failed")
            return result
        
        # Save results.objects to a file
        with open('results.json', 'w') as f:
            f.write(result.objects)
        


        # --- Step 5: Object Validation ---
        # logger.debug("Starting object validation")
        # visualization_image = base64.b64decode(result.visualization)
        
        # validated_objects = await vision_processor.validate_detections(
        #     visualization_image,
        #     [obj.model_dump() for obj in result.objects]
        # )

        # if not validated_objects:
        #     logger.error("No valid detections after validation")
        #     return ProcessingResponse(
        #         status=ProcessingStatus.ERROR,
        #         message="No valid detections after validation"
        #     )
        # logger.info(f"Object validation completed - Valid objects: {len(validated_objects)}")

        # # --- Step 6: Object Classification ---
        # logger.debug("Starting object classification")
        # image_objects = [
        #     obj for obj in validated_objects 
        #     if obj["layer_type"] == LayerType.IMAGE
        # ]
        # other_objects = [
        #     obj for obj in validated_objects 
        #     if obj["layer_type"] != LayerType.IMAGE
        # ]
        # logger.info(f"Objects classified - Images: {len(image_objects)}, Other: {len(other_objects)}")

        # # --- Step 7: Style Enhancement ---
        # if other_objects:
        #     logger.debug("Starting style enhancement")
        #     enhanced_data = await vision_processor.enhance_styles(
        #         visualization_image,
        #         other_objects
        #     )
        #     logger.info("Style enhancement completed successfully")
        # else:
        #     logger.info("Skipping style enhancement - no non-image objects")
        #     enhanced_data = {"elements": []}

        # # --- Step 8: Result Assembly ---
        # logger.debug("Assembling final results")
        # result.objects = [
        #     DetectedObject(**obj) 
        #     for obj in enhanced_data["elements"]
        # ]
        # result.objects.extend([
        #     DetectedObject(**obj) 
        #     for obj in image_objects
        # ])

        # # --- Step 9: Output Generation ---
        # logger.debug("Generating outputs")
        # result = globals_container.processor.regenerate_outputs(
        #     image_content=image_content,
        #     validated_objects=result.objects
        # )

        # # --- Step 10: Inpainting ---
        # logger.debug("Starting inpainting process")
        # inpaint_client = InpaintAPIClient()
        # result.inpainted_image = await inpaint_client.inpaint_image(
        #     original_image_url=result.original_image,
        #     mask_image_url=result.masked_output
        # )
        # logger.info("Inpainting completed successfully")

        # # --- Step 11: Usage Statistics ---
        # result.usage = {
        #     "prompt_generation": prompt_result.get("usage", {}),
        #     "vision_processing": vision_processor.get_total_usage(),
        #     "image_analysis": analysis_result.get("usage", {})
        # }
        # logger.info("Processing completed successfully")

        return result

    except Exception as e:
        logger.error(f"Error processing image: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@app.get("/api/v2/health")
async def health_check():
    logger.debug("Health check requested")
    status = {
        "status": "healthy",
        "version": app.version,
        "device": str(globals_container.processor.device if globals_container.processor else "not initialized"),
        "model_manager_status": "initialized" if globals_container.model_manager else "not initialized",
        "processor_status": "initialized" if globals_container.processor else "not initialized"
    }
    logger.info(f"Health check completed - Status: {status['status']}")
    return status

@app.post("/api/v2/cleanup")
async def force_cleanup():
    logger.info("Manual cleanup requested")
    try:
        if globals_container.processor:
            globals_container.processor.cleanup_resources()
            logger.info("Processor resources cleaned up successfully")
        if globals_container.model_manager:
            globals_container.model_manager.cleanup()
            logger.info("Model manager cleaned up successfully")
        return {"status": "success", "message": "Cleanup completed successfully"}
    except Exception as e:
        logger.error(f"Manual cleanup failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    logger.error(f"HTTP exception occurred: {exc.detail} (Status: {exc.status_code})")
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
    logger.error(f"Unhandled exception occurred: {str(exc)}", exc_info=True)
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
    
    logger.info("Starting server...")
    uvicorn_config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=8097,
        log_level="info",
        timeout_keep_alive=30,
        limit_concurrency=10,
        timeout_notify=30,
        timeout_graceful_shutdown=30
    )
    
    server = uvicorn.Server(uvicorn_config)
    server.run()