# File: app/core/prompt_processor.py

from openai import AsyncOpenAI
from typing import Dict, Any
import os
import json

class PromptProcessor:
    """Prompt processor class for generating optimized image generation prompts using OpenAI API."""
    
    def __init__(self, api_key: str = None):
        """Initialize prompt processor with OpenAI API key."""
        self.client = AsyncOpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.total_usage = {
            "total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "prompt_cached_tokens": 0,
        }

    def _update_usage(self, response) -> Dict:
        """Update total token usage and return current usage statistics."""
        current_usage = {
            "total_tokens": response.usage.total_tokens,
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "prompt_cached_tokens": response.usage.prompt_tokens_details.cached_tokens
        }
        
        # Update total usage
        self.total_usage["total_tokens"] += current_usage["total_tokens"]
        self.total_usage["prompt_tokens"] += current_usage["prompt_tokens"]
        self.total_usage["completion_tokens"] += current_usage["completion_tokens"]
        self.total_usage["prompt_cached_tokens"] += current_usage["prompt_cached_tokens"]
        
        return current_usage

    def get_total_usage(self) -> Dict:
        """Get total token usage across all operations."""
        return self.total_usage

    async def generate_prompt(
        self,
        base_prompt: str,
        tone: str,
        style: str,
        brand_name: str,
        selected_colors: list[str]
    ) -> Dict[str, Any]:
        """
        Generate an optimized image generation prompt based on input parameters.
        
        Args:
            base_prompt: Initial prompt text
            tone: Desired tone (e.g., "Casual", "Professional")
            style: Visual style (e.g., "Animated", "Realistic")
            brand_name: Name of the brand
            selected_colors: List of hex color codes
            
        Returns:
            Dictionary containing generated prompt and any additional parameters
        """
        # Convert hex colors to a more descriptive format
        color_descriptions = [f"#{color}" for color in selected_colors]
        color_list = ", ".join(color_descriptions)
        
        system_prompt = f"""
        Create an optimized image generation prompt based on the following criteria:

        Base Prompt: Create an [theme] modern social media advertisement for a [brand_type] Brand. A small brand logo is
        visible in the corner of the image. The brand name '[brand_name]' can be seen with an elegant font. A brand tagline in eye-catching font reads
        [brand_tagline]. A call-to-action button with the text [cta_text] is placed at the bottom of the image.
        The [procuct_type] is primary focus, positioned prominently. The background is [background].
        [theme].

        Fill the macros (all text within square brackets) with the appropriate details based on the user input.

        Additional Criteria:
        - [background] - Analyze the user input to determine the appropriate background for the image. Describe it in a visually appealing way.
        - [theme] - Analyze the user input to determine the theme of the image.
        - [brand_type] - Analyze the user input to determine the type of brand (e.g., fashion, tech, food).
        - [brand_name] - Use the brand name provided by the user.
        - [brand_tagline] - Analyze the user input to determine a suitable tagline for the brand.
        - [cta_text] - Analyze the user input to determine the call-to-action text.
        - [product_type] - Analyze the user input to determine the type of product being advertised. Describe it in a visually appealing way.
        - Use the brand colors (dont add hex codes directly, describe the colors) to create a visually appealing final image.
        """
        response = await self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": f"""Create an optimized image generation prompt based on:
                    User Prompt: {base_prompt}
                    Desired Tone: {tone}
                    Visual Style: {style}
                    Brand Name: {brand_name}
                    Brand Colors: {color_list}"""
                }
            ],
            max_tokens=2000,
            temperature=0.5
        )
        generated_prompt = response.choices[0].message.content.strip()
        
        # Update token usage
        usage_stats = self._update_usage(response)
        
        return {
            "prompt": generated_prompt,
            "usage": usage_stats
        }