"""Stateless Model Context Protocol support for the existing HTTP service."""

import base64
import json

from api_docs import CLONE_SCHEMA, DESIGN_SCHEMA
from config import DEFAULT_LANGUAGE, DEFAULT_MAX_WORDS, DEFAULT_MIN_WORDS, LANGUAGES
import voice_library

PROTOCOL_VERSION = "2025-03-26"
SUPPORTED_PROTOCOLS = {"2024-11-05", "2025-03-26"}


class MCPService:
    def __init__(self, api_service):
        self.api = api_service

    @staticmethod
    def _response(identifier, result=None, error=None):
        response = {"jsonrpc": "2.0", "id": identifier}
        if error is not None:
            response["error"] = error
        else:
            response["result"] = result
        return response

    @staticmethod
    def _tools():
        return [
            {"name": "list_voices", "description": "List saved Qwen3-TTS voices and metadata.",
             "inputSchema": {"type": "object", "properties": {}}},
            {"name": "get_config", "description": "Get supported languages and generation defaults.",
             "inputSchema": {"type": "object", "properties": {}}},
            {"name": "voice_design", "description": "Generate WAV audio from a voice description.",
             "inputSchema": DESIGN_SCHEMA},
            {"name": "voice_clone", "description": "Generate WAV speech with a saved voice or reference audio.",
             "inputSchema": CLONE_SCHEMA},
        ]

    def handle(self, message):
        """Return (HTTP status, JSON-RPC payload), or (202, None) for notifications."""
        if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
            return 400, self._response(None, error={"code": -32600, "message": "Invalid JSON-RPC request."})
        identifier = message.get("id")
        method = message.get("method")
        if identifier is None:
            return 202, None
        params = message.get("params") or {}
        if not isinstance(params, dict):
            return 200, self._response(identifier, error={"code": -32602, "message": "Parameters must be an object."})
        if method == "initialize":
            requested = params.get("protocolVersion")
            version = requested if requested in SUPPORTED_PROTOCOLS else PROTOCOL_VERSION
            return 200, self._response(identifier, {
                "protocolVersion": version,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "qwen3-tts", "version": "1.0.0"},
                "instructions": "Use voice_design or voice_clone to generate WAV audio. Audio is base64-encoded; text content includes the saved output path.",
            })
        if method == "ping":
            return 200, self._response(identifier, {})
        if method == "tools/list":
            return 200, self._response(identifier, {"tools": self._tools()})
        if method == "tools/call":
            return self._call(identifier, params)
        return 200, self._response(identifier, error={"code": -32601, "message": "Method not found."})

    def _call(self, identifier, params):
        name = params.get("name")
        arguments = params.get("arguments") or {}
        if not isinstance(arguments, dict):
            return 200, self._response(identifier, error={"code": -32602, "message": "Tool arguments must be an object."})
        try:
            if name == "list_voices":
                voices = [{k: v for k, v in item.items() if k != "path"}
                          for item in voice_library.list_voice_details()]
                return 200, self._response(identifier, _text_result({"voices": voices}))
            if name == "get_config":
                config = {"languages": LANGUAGES, "defaults": {
                    "language": DEFAULT_LANGUAGE, "min_words": int(DEFAULT_MIN_WORDS),
                    "max_words": int(DEFAULT_MAX_WORDS), "temperature": 0.7}}
                return 200, self._response(identifier, _text_result(config))
            if name in ("voice_design", "voice_clone"):
                method = self.api.design if name == "voice_design" else self.api.clone
                return 200, self._response(identifier, _audio_result(method(arguments)))
        except Exception as exc:
            return 200, self._response(identifier, {"content": [{"type": "text", "text": str(exc)}],
                                                    "isError": True})
        return 200, self._response(identifier, error={"code": -32602, "message": "Unknown tool."})


def _text_result(value):
    return {"content": [{"type": "text", "text": json.dumps(value, ensure_ascii=False)}]}


def _audio_result(result):
    metadata = {key: value for key, value in result.items() if key != "audio"}
    return {"content": [
        {"type": "text", "text": json.dumps(metadata, ensure_ascii=False)},
        {"type": "audio", "data": base64.b64encode(result["audio"]).decode("ascii"),
         "mimeType": "audio/wav"},
    ]}
