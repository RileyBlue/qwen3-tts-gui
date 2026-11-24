#!/usr/bin/env python3
"""Entry point for the Qwen3-TTS application.

Usage:
    python launch.py                  # Launch the GUI and localhost API
    python launch.py --api-only       # Launch only the localhost API
    python launch.py test             # Run unit tests
    python launch.py benchmark --voice NAME --text-file PATH
"""

import argparse
import sys


def _runtime_parser():
    from config import ATTNS, DEFAULT_ATTN, DEFAULT_DEVICE, DEFAULT_DTYPE, DTYPES
    parser = argparse.ArgumentParser(description="Launch the Qwen3-TTS GUI and HTTP API.")
    parser.add_argument("--api-only", action="store_true",
                        help="run only the HTTP API, without importing or opening the GUI")
    parser.add_argument("--no-api", action="store_true", help="disable the API in GUI mode")
    parser.add_argument("--host", default="127.0.0.1", help="API bind address (default: localhost)")
    parser.add_argument("--port", type=int, default=8765, help="API port (default: 8765)")
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="headless model device")
    parser.add_argument("--dtype", choices=DTYPES, default=DEFAULT_DTYPE, help="headless model dtype")
    parser.add_argument("--attn", choices=ATTNS, default=DEFAULT_ATTN, help="headless attention implementation")
    return parser


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        from tests import run_tests
        run_tests()
    elif len(sys.argv) > 1 and sys.argv[1] == "benchmark":
        from benchmark import main as run_benchmark
        raise SystemExit(run_benchmark(sys.argv[2:]))
    else:
        args = _runtime_parser().parse_args()
        if args.api_only and args.no_api:
            _runtime_parser().error("--api-only and --no-api cannot be used together")
        if args.api_only:
            from api_server import run_api
            run_api(args.host, args.port, args.device, args.dtype, args.attn)
        else:
            from qt_app import main as run_gui
            run_gui(api_enabled=not args.no_api, api_host=args.host, api_port=args.port)


if __name__ == "__main__":
    main()
