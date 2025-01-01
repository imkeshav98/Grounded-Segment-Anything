# File: app/utils/inpaint.py

import aiohttp
import json
from typing import Optional
from pathlib import Path
import os
from dotenv import load_dotenv

class InpaintAPIClient:
    """Client for interacting with the Inpaint API service"""
    
    def __init__(self, base_url: Optional[str] = None):
        """
        Initialize the Inpaint API client
        
        Args:
            base_url: Optional base URL for the API. If not provided, uses environment variable
        """
        load_dotenv()
        self.base_url = (base_url or os.getenv("INPAINT_API_URL", "https://inpaint-api.adbox.pro")).rstrip("/")

    async def inpaint_image(
        self,
        original_image_url: str,
        mask_image_url: str,
        expand_mask: int = 30
    ) -> str:
        """
        Perform inpainting on an image using the API
        
        Args:
            original_image_url: URL of the original image
            mask_image_url: URL of the mask image
            expand_mask: Number of pixels to expand the mask by (default: 30)
            
        Returns:
            URL of the inpainted result image
            
        Raises:
            Exception: If the API request fails
        """
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/inpaint",
                json={
                    "original_image_url": original_image_url,
                    "mask_image_url": mask_image_url,
                    "expand_mask": expand_mask
                },
                headers={"Content-Type": "application/json"}
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Inpaint API error ({response.status}): {error_text}")
                
                data = await response.json()
                return data["result"]