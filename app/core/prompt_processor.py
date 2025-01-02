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
        
        system_prompt = f"""You are an expert in creating prompts for generating professional advertising banners and marketing visuals. 
        Your task is to enhance the given prompt (paragraph within 150 words) specifically for advertisement banner generation while incorporating:

        - Ensure the prompt creates a compelling advertising visual
        - Incorporate 'Brand Name' and 'Brand Colors' into the prompt
        - Use brand colors effectively
        - Google, Facebook, and Instagram audience targeting
        - Product or service should be shown effectively

         Style and Tone:
           - Visual Style: Output should match the user's selected style preference
           - Communication Tone: Ensure the prompt reflects the desired tone.
           - Make sure the output aligns with the brand's image and message

        Technical Specifications:
           - Ensure clear space for text overlay
           - High-quality, commercial-grade output
           - Image should be suitable for real marketing campaigns
           - Final output should be optimized for advertising purposes
           - Must have a call to action Button

        CRITICAL:
            - Final prompt should match the style and tone of the brand
            - Output should be a paragraph and within 150 words

        Keep the output focused on creating an effective advertising banner that could be used in real marketing campaigns."""

        response = await self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": f"""Create an optimized image generation prompt based on: {base_prompt}
                    Desired Tone: {tone}
                    Visual Style: {style}
                    Brand Name: {brand_name}
                    Brand Colors: {color_list}"""
                }
            ],
            max_tokens=2000,
            temperature=0.7
        )
        
        generated_prompt = response.choices[0].message.content.strip()
        
        # Update token usage
        usage_stats = self._update_usage(response)
        
        return {
            "prompt": generated_prompt,
            "usage": usage_stats
        }