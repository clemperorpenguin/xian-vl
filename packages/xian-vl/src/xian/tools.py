from typing import Any

# Standard OpenAI tool definitions for the Lemonade OmniRouter

OMNI_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "generate_image",
            "description": "Generate a new image from a text prompt using an image generation model.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "A detailed description of the image to generate.",
                    },
                    "size": {
                        "type": "string",
                        "description": "The size of the generated image.",
                        "default": "1024x1024",
                        "enum": ["512x512", "768x768", "1024x1024"]
                    }
                },
                "required": ["prompt"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_image",
            "description": "Modify an existing image based on a text prompt.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "What to add, remove, or modify in the image.",
                    },
                    "image_path": {
                        "type": "string",
                        "description": "The local file path of the image to edit, or 'screenshot'/'image' to modify the current screen capture.",
                    }
                },
                "required": ["prompt", "image_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "text_to_speech",
            "description": "Convert text into spoken audio (speech synthesis) and play it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The text content to convert to speech.",
                    },
                    "voice": {
                        "type": "string",
                        "description": "The voice to use for narration.",
                        "default": "af_heart"
                    }
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "transcribe_audio",
            "description": "Transcribe speech from an audio file to text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "audio_path": {
                        "type": "string",
                        "description": "The path to the audio file (WAV or MP3).",
                    }
                },
                "required": ["audio_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_image",
            "description": "Analyze an image or screenshot to answer questions about its content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "string",
                        "description": "The local file path of the image/screenshot to analyze, or 'screenshot'/'image' to analyze the current screen.",
                    },
                    "question": {
                        "type": "string",
                        "description": "What you want to know about the image.",
                    }
                },
                "required": ["image_path", "question"],
            },
        },
    }
]
