# File: app/core/vision_processor.py

from app.models.vision_schemas import VALIDATION_SCHEMA, ADVERTISEMENT_SCHEMA
import base64
from openai import AsyncOpenAI
from typing import List, Dict, Any
import os
import json

class VisionProcessor:
    """Vision processor class for image analysis and enhancement using OpenAI API."""
    
    def __init__(self, api_key: str = None):
        """Initialize vision processor with OpenAI API key."""
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

    async def analyze_image(self, image_content: bytes, user_prompt: str) -> Dict[str, Any]:
        """
        Generate initial prompt from image analysis.
        
        Args:
            image_content: Raw image bytes
            
        Returns:
            Dictionary containing detected items, prompt, and token usage
        """
        base64_image = base64.b64encode(image_content).decode('utf-8')
        
        response = await self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional designer specialized in advertisement analysis. Focus on identifying prominent foreground elements and interactive components while ignoring background elements."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "high"
                            },
                        },
                        {
                            "type": "text",
                            "text": f"""Analyze this Advertisement image and identify only the foreground visual elements.

                            Output Format: A string with . separated values for each prominent element detected. Example: "Clickable UI button. Person. Car."
                            
                            Guidelines for detection:
                            1. Ignore all text elements and typography
                            2. Exclude background imagery like landscapes, patterns, or decorative elements
                            3. Identify interactive UI elements as "Clickable UI button"
                            4. Focus on main subjects and featured items:
                                - Primary product being advertised
                                - People and their notable attributes (clothing)
                                - Prominent objects directly related to the ad message
                                - Vehicles or transportation
                            6. For products, be specific about their category (e.g., "Smartphone device" instead of just "Device")
                            7. Check for Brand Logo carefully in Image. If detected, include it as "Logo."
                            
                            Hints:
                            - Look for elements that stand out visually or are interactive
                            - Elements that smaller or less detailed are likely background elements (e.g., trees, cloud, sky)
                            - The elements which repeats multiple times are likely background elements
                            - Use the user_prompt thats been used to generate the image for context.
                            - Person wearing a dress can be detected as a Person. No need to idetify as Person. Dress.

                            user_prompt: {user_prompt}
                            Notes: 
                            - Return the detected elements in a string separated by a period (.)
                            - All detected elements should be singular and start with a capital letter. Example: "Person. Car. Smartphone."
                            """
                        }
                    ]
                }
            ],
            max_tokens=1500
        )
        
        prompt_text = response.choices[0].message.content.strip()
        
        # Update total usage
        self._update_usage(response)
        return {
            "detectedItems": prompt_text.split(". "),
            "prompt": prompt_text
        }

    async def validate_detections(self, visualization_image: bytes, detections: List[Dict]) -> List[Dict]:
        """
        Validate detections using visualization.
        
        Args:
            visualization_image: Image bytes with visualized detections
            detections: List of detected objects
            
        Returns:
            List of validated detections
        """
        base64_image = base64.b64encode(visualization_image).decode('utf-8')
        
        response = await self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": """You are a professional designer and a professional reviewer who makes no mistake. Validate the detections in the image and return the valid detections in the json format.:

                    Context:
                    - The image shows a visualization of detected elements in an advertisement with bounding box, detected text, and Id.
                    - The JSON object contains the detected elements with their object_id and detected_text also.
                    - Your task is to validate the detections and mark them as valid or invalid. Example a Bag detected as a Clickable UI button is invalid
                      A Clickable UI button detected as a text is invalid. A Clickable UI button detected as a Clickable UI button is valid. A Person detected as a Person is valid.
                    - If the detection is invalid, provide object_id, is_valid as false and reason for invalidation and layer_type (button, text, image).
                    - Response should always follow JSON schema.

                    Hint:
                    - Always check the object_id and object for validation.
                    - Always provide layer_Type for valid detections
                    - Texts are mostly valid detections. But sometimes, they can be invalid. Example
                    - A text inside a button is not a valid text. The clickable UI button is a valid button with text.
                    - Logo may/ may not have text in it. Both are valid detections. But a text inside a logo is not a valid text.

                    Critical:
                    - Object_id should not be changed.
                    - Layer_type should be provided for valid detections.
                    - Always triple check the detections before submitting [CRITICAL].
                    """
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "high"
                            }
                        },
                        {
                            "type": "text",
                            "text": f"Properly validate these detections:\n{json.dumps(detections, indent=1)}"
                        }
                    ]
                }
            ],
            response_format={"type": "json_object"},
            functions=[{"name": "validate_detections", "parameters": VALIDATION_SCHEMA}],
            function_call={"name": "validate_detections"},
            max_tokens=4000,
        )
        
        validations = json.loads(response.choices[0].message.function_call.arguments)

        valid_detections = [
            {**det, "layer_type": next((v["layer_type"] for v in validations["valid_detections"] if v["object_id"] == det["object_id"]), None)}
            for det in detections 
            if any(v["object_id"] == det["object_id"] and v["is_valid"] for v in validations["valid_detections"])
        ]
        
        # Update total usage
        self._update_usage(response)
        return valid_detections

    async def enhance_styles(self, original_image: bytes, validated_objects: List[Dict]) -> Dict[str, Any]:
        """
        Enhance validated detections with styles.
        
        Args:
            original_image: Original image bytes
            validated_objects: List of validated objects
            
        Returns:
            Enhanced styles for validated objects
        """
        base64_image = base64.b64encode(original_image).decode('utf-8')
        
        response = await self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": """For each validated element, determine following and return in json format:
                    1. Exact Google Font matches. If not available, provide the closest match.
                    2. Font properties (size, weight, style)
                    3. Precise Yext and Button Background colors (hex codes)
                    4. Button properties (border radius, background color, text color)
                    5. Fix any text spelling errors if any or grammatical errors.The text should be in English and should be grammatically correct. Also extract overall theme colors and typography.
                    
                    CRITICAL: 
                    1. NEVER Change any BBOX COORDINATES ( x, y, width, height) of the elements.
                    2. NEVER Change the object_id of the elements.
                    3. NEVER Change Line_count and Alignment of text.
                    4. FONT FAMILY should be VALID GOOGLE FONT FAMILY.
                    5. ALWAYS TRIPPLE CHECK the styles AND ctiical details before submitting.
                    """
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "high"
                            }
                        },
                        {
                            "type": "text",
                            "text": f"Add styles to these elements:\n{json.dumps(validated_objects, indent=1)}. Make sure match the styling with the image."
                        }
                    ]
                }
            ],
            response_format={"type": "json_object"},
            functions=[{"name": "enhance_styles", "parameters": ADVERTISEMENT_SCHEMA}],
            function_call={"name": "enhance_styles"},
            max_tokens=4000
        )
        
        enhanced_data = json.loads(response.choices[0].message.function_call.arguments)

        # Update total usage
        self._update_usage(response)
        return enhanced_data