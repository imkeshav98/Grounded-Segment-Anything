# File: app/core/replicate_client.py

import os
import replicate
from typing import List
from dotenv import load_dotenv

class ReplicateClient:
    """Client for interacting with Replicate's image generation API"""
    
    def __init__(self):
        """Initialize Replicate client"""
        load_dotenv()
        api_token = os.getenv("REPLICATE_API_TOKEN")
        if not api_token:
            raise ValueError("REPLICATE_API_TOKEN is required")
            
        self.client = replicate.Client(api_token=api_token)

    async def generate_image(self, prompt: str) -> List[str]:
        """
        Generate image using Replicate's black-forest-labs/flux-dev model
        
        Args:
            prompt: The image generation prompt
            
        Returns:
            List of image URLs
        """
        try:
            response = self.client.run(
                "black-forest-labs/flux-dev",
                input={
                    "prompt": prompt,
                    "go_fast": False,
                    "guidance": 3.5,
                    "num_outputs": 1,
                    "aspect_ratio": "1:1",
                    "output_format": "png",
                    "output_quality": 100,
                    "prompt_strength": 0.8,
                    "num_inference_steps": 40
                }
            )
            
            return response["output"]

        except Exception as e:
            raise Exception(f"Image generation failed: {str(e)}")