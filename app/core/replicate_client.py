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

    def generate_image(self, prompt: str) -> List[bytes]:
        """
        Generate image using Replicate's black-forest-labs/flux-dev model
        
        Args:
            prompt: The image generation prompt
            
        Returns:
            List of bytes objects containing the generated images
        """
        try:
            output = self.client.run(
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
                    "num_inference_steps": 50
                }
            )

            # Always return a list of bytes for consistency
            return [item.read() for item in output]

        except Exception as e:
            raise Exception(f"Image generation failed: {str(e)}")