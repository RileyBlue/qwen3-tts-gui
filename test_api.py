"""Tests for the HTTP API using small fake models (no model downloads)."""

import base64
import json
import os
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from unittest.mock import patch

import numpy as np

from api_server import APIService, QwenAPIServer


class FakeModel:
    def generate_voice_design(self, text, language, instruct):
        return [np.full(240, 0.1, dtype=np.float32)], 24000

    def create_voice_clone_prompt(self, **kwargs):
        return ["prompt"]

    def generate_voice_clone(self, text, language, voice_clone_prompt, **kwargs):
        return [np.full(480, 0.2, dtype=np.float32)], 24000


class FakeModels:
    def __init__(self):
        self.model = FakeModel()
        self.generation_lock = threading.RLock()

    def get(self, kind, notify=None):
        return self.model


class APITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.service = APIService(FakeModels(), output_dir=self.temp.name)
        self.server = QwenAPIServer("127.0.0.1", 0, self.service)
        self.server.start()

    def tearDown(self):
        self.server.stop()
        self.temp.cleanup()

    def request(self, path, data=None):
        body = None if data is None else json.dumps(data).encode()
        request = urllib.request.Request(self.server.address + path, data=body)
        if body:
            request.add_header("Content-Type", "application/json")
        try:
            response = urllib.request.urlopen(request, timeout=5)
        except urllib.error.HTTPError as exc:
            response = exc
        with response:
            return response.status, dict(response.headers), response.read()

    def test_health_and_config(self):
        status, _, body = self.request("/health")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["status"], "ok")
        status, _, body = self.request("/v1/config")
        self.assertEqual(status, 200)
        self.assertIn("english", json.loads(body)["languages"])

    def test_design_returns_wav_and_saves_output(self):
        payload = {"description": "warm narrator", "text": "Hello", "seed": 7}
        status, headers, body = self.request("/v1/tts/design", payload)
        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "audio/wav")
        self.assertEqual(headers["X-Seed"], "7")
        self.assertTrue(body.startswith(b"RIFF"))
        self.assertTrue(os.path.isfile(headers["X-Output-Path"]))

    @patch("voice_library.load_voice")
    def test_clone_with_saved_voice(self, load_voice):
        load_voice.return_value = ({"description": "warm", "seed": 7}, ["prompt"], "/unused")
        payload = {"voice": "Narrator", "text": "Hello world", "seed": 9}
        status, headers, body = self.request("/v1/tts/clone", payload)
        self.assertEqual(status, 200)
        self.assertEqual(headers["X-Sample-Rate"], "24000")
        self.assertTrue(body.startswith(b"RIFF"))
        load_voice.assert_called_once_with("Narrator")

    def test_invalid_requests_and_busy_response(self):
        status, _, body = self.request("/v1/tts/design", {"description": ""})
        self.assertEqual(status, 400)
        self.assertIn("description", json.loads(body)["error"])
        status, _, _ = self.request("/v1/tts/design", {"description": "x", "language": 3})
        self.assertEqual(status, 400)
        self.service.gui_busy = lambda: True
        status, _, body = self.request("/v1/tts/design", {"description": "x"})
        self.assertEqual(status, 409)
        self.assertIn("GUI", json.loads(body)["error"])

    def test_openapi_and_swagger(self):
        status, headers, body = self.request("/docs")
        self.assertEqual(status, 200)
        self.assertIn("text/html", headers["Content-Type"])
        self.assertIn(b"SwaggerUIBundle", body)
        status, _, body = self.request("/openapi.json")
        spec = json.loads(body)
        self.assertEqual(status, 200)
        self.assertEqual(spec["openapi"], "3.0.3")
        self.assertIn("/v1/tts/clone", spec["paths"])
        self.assertIn("/mcp", spec["paths"])

    def test_mcp_initialize_and_tool_discovery(self):
        status, _, body = self.request("/mcp", {
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"protocolVersion": "2025-03-26", "capabilities": {}, "clientInfo": {}}})
        result = json.loads(body)["result"]
        self.assertEqual(status, 200)
        self.assertEqual(result["protocolVersion"], "2025-03-26")
        status, _, body = self.request("/mcp", {
            "jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
        tools = json.loads(body)["result"]["tools"]
        self.assertEqual({tool["name"] for tool in tools},
                         {"list_voices", "get_config", "voice_design", "voice_clone"})

    def test_mcp_generation_returns_audio_content(self):
        status, _, body = self.request("/mcp", {
            "jsonrpc": "2.0", "id": 3, "method": "tools/call",
            "params": {"name": "voice_design", "arguments": {
                "description": "warm narrator", "text": "Hello", "seed": 7}}})
        content = json.loads(body)["result"]["content"]
        self.assertEqual(status, 200)
        self.assertEqual(content[1]["type"], "audio")
        self.assertTrue(base64.b64decode(content[1]["data"]).startswith(b"RIFF"))

    def test_unknown_route(self):
        status, _, _ = self.request("/missing")
        self.assertEqual(status, 404)


if __name__ == "__main__":
    unittest.main()
