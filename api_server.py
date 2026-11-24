"""Dependency-free HTTP API for GUI and headless Qwen3-TTS use."""

import json
import os
import random
import threading
import uuid
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from api_docs import SWAGGER_HTML, openapi_document
from audio_utils import save_outputs
from config import (DEFAULT_ATTN, DEFAULT_DEVICE, DEFAULT_DTYPE, DEFAULT_LANGUAGE,
                    DEFAULT_MAX_WORDS, DEFAULT_MIN_WORDS, LANGUAGES, OUTPUT_DIR)
from model_loader import ModelManager
from mcp_api import MCPService, PROTOCOL_VERSION as MCP_PROTOCOL_VERSION
from tts_generator import TTSGenerator
import voice_library


class BusyError(RuntimeError):
    pass


class APIService:
    """Validate requests, run generation, and persist API audio outputs."""

    def __init__(self, models, output_dir=OUTPUT_DIR, gui_busy=None):
        self.models = models
        self.output_dir = output_dir
        self.gui_busy = gui_busy or (lambda: False)
        self._lock = threading.Lock()
        os.makedirs(output_dir, exist_ok=True)

    @property
    def busy(self):
        return self._lock.locked() or bool(self.gui_busy())

    @staticmethod
    def _text(data, key="text"):
        value = data.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"'{key}' must be a non-empty string.")
        return value.strip()

    @staticmethod
    def _language(data):
        language = data.get("language", DEFAULT_LANGUAGE)
        if not isinstance(language, str):
            raise ValueError("'language' must be a string.")
        language = language.lower()
        if language not in LANGUAGES:
            raise ValueError(f"Unsupported language '{language}'. See GET /v1/config.")
        return language

    @staticmethod
    def _integer(data, key, default=None, minimum=None):
        value = data.get(key, default)
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"'{key}' must be an integer or null.")
        if minimum is not None and value < minimum:
            raise ValueError(f"'{key}' must be at least {minimum}.")
        return value

    def _path(self, prefix):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(self.output_dir, f"API_{prefix}_{stamp}_{uuid.uuid4().hex[:8]}.wav")

    def _run(self, work):
        if self.gui_busy():
            raise BusyError("The GUI is running another generation; retry when it finishes.")
        if not self._lock.acquire(blocking=False):
            raise BusyError("The API is running another generation; retry when it finishes.")
        try:
            return work()
        finally:
            self._lock.release()

    def design(self, data):
        description = self._text(data, "description")
        text = self._text(data) if data.get("text") is not None else None
        language = self._language(data)
        seed = self._integer(data, "seed", random.randint(0, 2**31 - 1), 0)

        def work():
            wav, sample_rate = TTSGenerator(self.models).design_sample(
                description, language, seed, text)
            path = self._path("VoiceDesign")
            save_outputs([wav], sample_rate, path)
            return path, sample_rate, seed
        return _run_result(self._run(work))

    def clone(self, data):
        text = self._text(data)
        language = self._language(data)
        seed = self._integer(data, "seed", None, 0)
        minimum = self._integer(data, "min_words", int(DEFAULT_MIN_WORDS), 0)
        maximum = self._integer(data, "max_words", int(DEFAULT_MAX_WORDS), 0)
        if minimum and maximum and minimum > maximum:
            raise ValueError("'min_words' cannot exceed 'max_words'.")
        for key in ("voice", "voice_name", "ref_audio", "ref_text", "reuse_prompt",
                    "save_prompt", "description"):
            if data.get(key) is not None and not isinstance(data[key], str):
                raise ValueError(f"'{key}' must be a string.")
        x_vector_only = data.get("x_vector_only", False)
        if not isinstance(x_vector_only, bool):
            raise ValueError("'x_vector_only' must be a boolean.")
        temperature = data.get("temperature", 0.7)
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
            raise ValueError("'temperature' must be a number.")
        if not 0 <= temperature <= 2:
            raise ValueError("'temperature' must be between 0 and 2.")

        def work():
            wavs, sample_rate = TTSGenerator(self.models).clone(
                text=text, language=language, min_words=minimum, max_words=maximum,
                seed=seed, temperature=float(temperature),
                voice_name=data.get("voice") or data.get("voice_name"),
                ref_audio=data.get("ref_audio", ""), ref_text=data.get("ref_text"),
                x_vector_only=x_vector_only,
                reuse_prompt=data.get("reuse_prompt"), save_prompt=data.get("save_prompt"),
                description=data.get("description", ""))
            if not wavs:
                raise RuntimeError("Generation stopped before producing audio.")
            path = self._path("VoiceClone")
            save_outputs(wavs, sample_rate, path)
            return path, sample_rate, seed
        return _run_result(self._run(work))


def _run_result(result):
    path, sample_rate, seed = result
    with open(path, "rb") as audio_file:
        audio = audio_file.read()
    return {"audio": audio, "path": path, "sample_rate": sample_rate,
            "seed": seed, "filename": os.path.basename(path)}


class QwenAPIHandler(BaseHTTPRequestHandler):
    service = None
    mcp = None
    server_version = "Qwen3TTS/1.0"

    def _json(self, status, payload, headers=None):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def _html(self, html):
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _empty(self, status):
        self.send_response(status)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _wav(self, result):
        audio = result["audio"]
        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(len(audio)))
        self.send_header("Content-Disposition", f'attachment; filename="{result["filename"]}"')
        self.send_header("X-Sample-Rate", str(result["sample_rate"]))
        self.send_header("X-Output-Path", result["path"])
        if result["seed"] is not None:
            self.send_header("X-Seed", str(result["seed"]))
        self.end_headers()
        self.wfile.write(audio)

    def do_GET(self):
        if self.path == "/docs":
            self._html(SWAGGER_HTML)
        elif self.path == "/openapi.json":
            self._json(200, openapi_document())
        elif self.path == "/mcp":
            self._json(405, {"error": "Server-initiated event streams are not supported; use POST."})
        elif self.path == "/health":
            self._json(200, {"status": "ok", "service": "qwen3-tts", "busy": self.service.busy})
        elif self.path == "/v1/voices":
            voices = [{k: v for k, v in item.items() if k != "path"}
                      for item in voice_library.list_voice_details()]
            self._json(200, {"voices": voices})
        elif self.path == "/v1/config":
            self._json(200, {"languages": LANGUAGES, "defaults": {
                "language": DEFAULT_LANGUAGE, "min_words": int(DEFAULT_MIN_WORDS),
                "max_words": int(DEFAULT_MAX_WORDS), "temperature": 0.7}})
        else:
            self._json(404, {"error": "Not found."})

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 1_000_000:
                raise ValueError("Request body must be between 1 byte and 1 MB.")
            data = json.loads(self.rfile.read(length))
            if not isinstance(data, dict):
                raise ValueError("The JSON request body must be an object.")
            if self.path == "/mcp":
                status, response = self.mcp.handle(data)
                headers = {"MCP-Protocol-Version": self.headers.get(
                    "MCP-Protocol-Version", MCP_PROTOCOL_VERSION)}
                if response is None:
                    self._empty(status)
                else:
                    self._json(status, response, headers)
            elif self.path == "/v1/tts/design":
                self._wav(self.service.design(data))
            elif self.path == "/v1/tts/clone":
                self._wav(self.service.clone(data))
            else:
                self._json(404, {"error": "Not found."})
        except (ValueError, json.JSONDecodeError) as exc:
            self._json(400, {"error": str(exc)})
        except BusyError as exc:
            self._json(409, {"error": str(exc)})
        except Exception as exc:
            self._json(500, {"error": str(exc)})

    def log_message(self, fmt, *args):
        print(f"API {self.client_address[0]} - {fmt % args}")


class QwenAPIServer:
    """HTTP server that can run in a GUI-owned background thread."""

    def __init__(self, host, port, service):
        self.service = service
        handler = type("BoundQwenAPIHandler", (QwenAPIHandler,),
                       {"service": service, "mcp": MCPService(service)})
        self.httpd = ThreadingHTTPServer((host, port), handler)
        self.httpd.daemon_threads = True
        self.thread = None

    @property
    def address(self):
        host, port = self.httpd.server_address[:2]
        return f"http://{host}:{port}"

    def start(self):
        if self.thread is not None:
            return
        self.thread = threading.Thread(target=self.httpd.serve_forever,
                                       name="qwen-tts-api", daemon=True)
        self.thread.start()

    def serve_forever(self):
        self.httpd.serve_forever()

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        if self.thread is not None:
            self.thread.join(timeout=5)
            self.thread = None


def run_api(host="127.0.0.1", port=8765, device=DEFAULT_DEVICE,
            dtype=DEFAULT_DTYPE, attn=DEFAULT_ATTN):
    """Run the API without importing or starting any GUI components."""
    settings = (device, dtype, attn)
    models = ModelManager(lambda: settings)
    server = QwenAPIServer(host, port, APIService(models))
    print(f"Qwen3-TTS API listening on {server.address} (Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.httpd.server_close()
