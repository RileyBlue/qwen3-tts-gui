"""OpenAPI and Swagger UI assets for the Qwen3-TTS HTTP API."""

from config import DEFAULT_LANGUAGE, DEFAULT_MAX_WORDS, DEFAULT_MIN_WORDS, LANGUAGES


def _generation_properties():
    return {
        "language": {"type": "string", "enum": LANGUAGES, "default": DEFAULT_LANGUAGE},
        "seed": {"type": "integer", "minimum": 0, "nullable": True},
        "temperature": {"type": "number", "minimum": 0, "maximum": 2, "default": 0.7},
        "min_words": {"type": "integer", "minimum": 0, "default": int(DEFAULT_MIN_WORDS)},
        "max_words": {"type": "integer", "minimum": 0, "default": int(DEFAULT_MAX_WORDS)},
    }


DESIGN_SCHEMA = {
    "type": "object",
    "required": ["description"],
    "properties": {
        "description": {"type": "string", "description": "VoiceDesign style and timbre instructions."},
        "text": {"type": "string", "description": "Optional audition text; uses the standard passage when omitted."},
        "language": _generation_properties()["language"],
        "seed": {"type": "integer", "minimum": 0, "description": "Random seed; generated when omitted."},
    },
}

CLONE_SCHEMA = {
    "type": "object",
    "required": ["text"],
    "properties": {
        "text": {"type": "string", "description": "Text to synthesize; [emotion] tags are supported."},
        "voice": {"type": "string", "description": "Saved voice name, as returned by list_voices."},
        "ref_audio": {"type": "string", "description": "Local reference-audio path when no saved voice is used."},
        "ref_text": {"type": "string", "description": "Exact transcript of ref_audio."},
        "reuse_prompt": {"type": "string", "description": "Local reusable clone-prompt path."},
        "save_prompt": {"type": "string", "description": "Optional path at which to save the clone prompt."},
        "x_vector_only": {"type": "boolean", "default": False},
        "description": {"type": "string", "description": "Voice description required for emotion tags with a raw prompt."},
        **_generation_properties(),
    },
}

SWAGGER_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Qwen3-TTS API</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css">
</head><body><div id="swagger-ui"></div>
<script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
<script>window.onload = () => SwaggerUIBundle({url: '/openapi.json', dom_id: '#swagger-ui',
deepLinking: true, displayRequestDuration: true, tryItOutEnabled: true});</script>
</body></html>"""


def _json_response(description="Successful response"):
    return {"description": description, "content": {"application/json": {
        "schema": {"type": "object"}}}}


def _generation_responses():
    return {
        "200": {"description": "Generated 16-bit PCM WAV audio.", "headers": {
            "X-Output-Path": {"schema": {"type": "string"}},
            "X-Sample-Rate": {"schema": {"type": "integer"}},
            "X-Seed": {"schema": {"type": "integer"}}},
            "content": {"audio/wav": {"schema": {"type": "string", "format": "binary"}}}},
        "400": _json_response("Invalid request."),
        "409": _json_response("Another GUI or API generation is running."),
        "500": _json_response("Generation failed."),
    }


def openapi_document():
    request_body = lambda schema: {"required": True, "content": {
        "application/json": {"schema": schema}}}
    return {
        "openapi": "3.0.3",
        "info": {"title": "Qwen3-TTS API", "version": "1.0.0",
                 "description": "Local VoiceDesign and VoiceClone generation. The MCP endpoint is /mcp."},
        "servers": [{"url": "/"}],
        "paths": {
            "/health": {"get": {"summary": "Server and generation status",
                                 "responses": {"200": _json_response()}}},
            "/v1/voices": {"get": {"summary": "List saved voices",
                                    "responses": {"200": _json_response()}}},
            "/v1/config": {"get": {"summary": "List languages and defaults",
                                    "responses": {"200": _json_response()}}},
            "/v1/tts/design": {"post": {"summary": "Generate a VoiceDesign sample",
                "requestBody": request_body(DESIGN_SCHEMA), "responses": _generation_responses()}},
            "/v1/tts/clone": {"post": {"summary": "Generate speech with VoiceClone",
                "requestBody": request_body(CLONE_SCHEMA), "responses": _generation_responses()}},
            "/mcp": {"post": {"summary": "Model Context Protocol JSON-RPC endpoint",
                "description": "Supports initialize, ping, tools/list, and tools/call.",
                "requestBody": request_body({"type": "object"}),
                "responses": {"200": _json_response(), "202": {"description": "Notification accepted."}}}},
        },
    }
