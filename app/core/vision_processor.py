# File: app/core/vision_processor.py

from .schemas import VALIDATION_SCHEMA, ADVERTISEMENT_SCHEMA
import base64
from openai import OpenAI
from typing import List, Dict, Any
import os
import json

class VisionProcessor:
    def __init__(self, api_key: str = None):
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    async def analyze_image(self, image_content: bytes) -> Dict[str, Any]:
        """Generate initial prompt"""
        base64_image = base64.b64encode(image_content).decode('utf-8')
        
        response = await self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional designer. Analyze this image and return the visual elements with good accuracy."
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
                            "text": """Analyze this Advertisement image and return the visual elements.

                            Output Format: A string with . separated values for each element detected. Example: "Clickable UI button. Person. Shoe. Car."
                            
                            Things to keep in mind:
                            1. No need to detect any text elements.
                            2. No need to detect any element which is part of advertisement image background.
                            3. Check for any clickable UI elements like buttons and return is as "Clickable UI button".
                            4. Figure out the main objects in the image and return them as separate elements.
                            """
                        }
                    ]
                }
            ],
            max_tokens=1500
        )
        
        # Get the string response and convert to expected format
        prompt_text = response.choices[0].message.content.strip()
        return {
            "detectedItems": prompt_text.split(". "),
            "prompt": prompt_text
        }

    async def validate_detections(self, visualization_image: bytes, detections: List[Dict]) -> List[Dict]:
        """Step 1: Validate detections using visualization"""
        base64_image = base64.b64encode(visualization_image).decode('utf-8')
        
        response = await self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": """You are a professional designer and a professional reviewer. Validate the detections in the image and return the valid detections in the JSON format.:

                    Context:
                    - The image shows a visualization of detected elements in an advertisement with bounding box, detected text, and Id.
                    - The JSON object contains the detected elements with their object_id and detected_text also.
                    - Your task is to validate the detections and mark them as valid or invalid. Example a Bag detected as a Clickable UI button is invalid
                      A Clickable UI button detected as a text is invalid. A Clickable UI button detected as a Clickable UI button is valid. A Person detected as a Person is valid.
                    - If the detection is invalid, provide object_id, is_valid as false and reason for invalidation.
                    - Response should always follow JSON schema.
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
                            "text": f"Validate these detections:\n{json.dumps(detections, indent=2)}"
                        }
                    ]
                }
            ],
            response_format={"type": "json_object"},
            functions=[{"name": "validate_detections", "parameters": VALIDATION_SCHEMA}],
            function_call={"name": "validate_detections"},
            max_tokens=3000,
        )
        
        validations = json.loads(response.choices[0].message.function_call.arguments)
        return [det for det in detections 
                if any(v["object_id"] == det["object_id"] and v["is_valid"] 
                      for v in validations["valid_detections"])]

    async def enhance_styles(self, original_image: bytes, validated_objects: List[Dict]) -> Dict[str, Any]:
        """Step 2: Enhance validated detections with styles"""
        base64_image = base64.b64encode(original_image).decode('utf-8')
        
        response = await self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": """For each validated element, determine:
                    1. Exact Google Font matches. If not available, provide the closest match.
                    2. Font properties (size, weight, style)
                    3. Precise colors (hex codes)
                    5. Button properties (border radius, background color, text color)
                    6. Fix any text spelling errors if any or grammatical errors.The text should be in English and should be grammatically correct.
                    Also extract overall theme colors and typography.
                    
                    
                    CRITICAL: 
                    1. NEVER Change any BBOX COORDINATES ( x, y, width, height) of the elements.
                    2. NEVER Change the object_id of the elements.
                    3. NEVER Change Line_count and Alignment of text.
                    4. FONT FAMILY should be VALID GOOGLE FONT FAMILY.
                    5. ALWAYS DOUBLE CHECK the styles AND ctiical details before submitting.
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
                            "text": f"Add styles to these elements:\n{json.dumps(validated_objects, indent=2)}"
                        }
                    ]
                }
            ],
            response_format={"type": "json_object"},
            functions=[{"name": "enhance_styles", "parameters": ADVERTISEMENT_SCHEMA}],
            function_call={"name": "enhance_styles"},
            max_tokens=3000
        )
        
        return json.loads(response.choices[0].message.function_call.arguments)