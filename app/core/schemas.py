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
                    },
                    "reason": {"type": "string"}
                },
                "required": ["object_id", "is_valid"]
            }
        }
    }
}

ADVERTISEMENT_SCHEMA = {
    "type": "object",
    "required": ["elements"],  # Theme is optional
    "properties": {
        "elements": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["object_id", "layer_type", "object", "bbox", "confidence"],
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
                                "default": "Arial"
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
                                    "enum": ["bold", "italic", "underline", "strikethrough"]
                                },
                                "default": []
                            },
                            "color": {
                                "type": "string",
                                "default": "#000000"
                            },
                            "backgroundColor": {
                                "type": "string"
                            },
                            "borderRadius": {
                                "type": "number",
                                "default": 0
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