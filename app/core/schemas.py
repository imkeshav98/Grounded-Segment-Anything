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
                    "layer_type": {
                        "type": "string",
                        "enum": ["button", "text", "image"],
                        "description": "Layer type of the object as detected in the image"
                    },
                    "reason": {"type": "string"}
                },
                "required": ["object_id", "is_valid", "layer_type"]
            }
        }
    }
}

ADVERTISEMENT_SCHEMA = {
    "type": "object",
    "required": ["elements", "theme"],
    "properties": {
        "elements": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["object_id", "layer_type", "object", "bbox", "confidence", "detected_text", "styles", "line_count", "text_alignment"],
                "properties": {
                    "object_id": {"type": "integer"},
                    "object": { "type": "string"},
                    "layer_type": {
                        "type": "string",
                        "enum": ["button", "text", "image"],
                    },
                    "bbox": {
                        "type": "object",
                        "required": ["x", "y", "width", "height"],
                        "properties": {
                            "x": {"type": "number"},
                            "y": {"type": "number"},
                            "width": {"type": "number"},
                            "height": {"type": "number"}
                        }
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0
                    },
                    "detected_text": {
                        "type": "string",
                        "default": ""
                    },
                    "text_alignment": {
                        "type": "string",
                        "enum": ["left", "center", "right"],
                        "default": "left"
                    },
                    "line_count": {
                        "type": "integer",
                        "default": 1
                    },
                    "styles": {
                        "type": "object",
                        "properties": {
                            "fontFamily": {
                                "type": "string",
                                "default": "Arial",
                                "description": "Font family for text or button as shown in the image, Must be a valid Google font family"
                            },
                            "fontSize": {
                                "type": "number",
                                "default": 16
                            },
                            "fontWeight": {
                                "type": "string",
                                "enum": ["300", "400", "500", "600", "700", "800"],
                                "default": "400"
                            },
                            "fontStyles": {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                    "enum": ["bold", "italic", "underline", "strikethrough"],
                                    "description": "Font styles for text"
                                },
                                "default": []
                            },
                            "color": {
                                "type": "string",
                                "default": "#000000",
                                "description" : "Exact color for text or button as shown in the image"
                            },
                            "backgroundColor": {
                                "type": "string",
                                "description" : "Exact Background color for text or button as shown in the image"
                            },
                            "borderRadius": {
                                "type": "number",
                                "default": 0,
                                "description" : "Border radius for buttons"
                            },
                        }
                    }
                }
            }
        },
        "theme": {
            "type": "object",
            "required": ["primaryColor", "secondaryColor", "backgroundColor", "fontStyles"],
            "properties": {
                "primaryColor": {
                    "type": "string",
                    "pattern": "^#([A-Fa-f0-9]{6})$"  # Enforce hex color format
                },
                "secondaryColor": {
                    "type": "string",
                    "pattern": "^#([A-Fa-f0-9]{6})$"
                },
                "backgroundColor": {
                    "type": "string",
                    "pattern": "^#([A-Fa-f0-9]{6})$"
                },
                "fontStyles": {
                    "type": "object",
                    "required": ["heading", "body"],
                    "properties": {
                        "heading": {"type": "string"},
                        "body": {"type": "string"}
                    }
                }
            }
        }
    }
}