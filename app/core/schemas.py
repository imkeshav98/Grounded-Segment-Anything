# File: app/core/schemas.py

VALIDATION_SCHEMA = {
    "type": "object",
    "properties": {
        "valid_detections": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "object_id": {"type": "number"},
                    "is_valid": {"type": "boolean"},
                    "reason": {"type": "string"}
                },
                "required": ["object_id", "is_valid"]
            }
        }
    }
}

ADVERTISEMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "elements": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "object_id": {"type": "number"},
                    "object": {"type": "string", "enum": ["button", "text", "image"]},
                    "bbox": {
                        "type": "object",
                        "properties": {
                            "x": {"type": "number"},
                            "y": {"type": "number"},
                            "width": {"type": "number"},
                            "height": {"type": "number"}
                        }
                    },
                    "no_of_lines": {"type": "number"},
                    "confidence": {"type": "number"},
                    "detected_text": {"type": "string"},
                    "styles": {
                        "type": "object",
                        "properties": {
                            "fontFamily": {"type": "string"},
                            "fontSize": {"type": "number"},
                            "fontWeight": {"type": "string"},
                            "fontStyles": {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                    "enum": ["bold", "italic", "underline", "strikethrough"]
                                }
                            },
                            "color": {"type": "string"},
                            "backgroundColor": {"type": "string"},
                            "borderRadius": {"type": "number"},
                            "alignment": {"type": "string"}
                        }
                    }
                }
            }
        },
        "theme": {
            "type": "object",
            "properties": {
                "primaryColor": {"type": "string"},
                "secondaryColor": {"type": "string"},
                "backgroundColor": {"type": "string"},
                "fontStyles": {
                    "type": "object",
                    "properties": {
                        "heading": {"type": "string"},
                        "body": {"type": "string"}
                    }
                }
            }
        }
    }
}