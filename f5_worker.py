"""Run F5-TTS in an optional separate Python environment."""

import atexit
import contextlib
import json
import os
import subprocess
import sys
import tempfile

import numpy as np


_PREFIX = "F5_RESULT:"


class F5Worker:
    def __init__(self, device):
        python = os.environ.get("F5_TTS_PYTHON") or sys.executable
        env = os.environ.copy()
        bin_dir = os.path.dirname(os.path.abspath(python))
        env["PATH"] = os.pathsep.join((bin_dir, os.path.join(bin_dir, "Library", "bin"), env.get("PATH", "")))
        try:
            self.process = subprocess.Popen(
                [python, "-u", os.path.abspath(__file__), device],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1, env=env,
            )
        except OSError as exc:
            raise RuntimeError(f"Cannot start F5-TTS Python interpreter: {python}") from exc
        atexit.register(self.close)
        try:
            self._read_result()
        except Exception:
            self.close()
            raise

    def _read_result(self):
        for line in self.process.stdout:
            if line.startswith(_PREFIX):
                result = json.loads(line[len(_PREFIX):])
                if not result.get("ok"):
                    raise RuntimeError(result.get("error", "F5-TTS worker failed."))
                return result
        raise RuntimeError("F5-TTS worker exited unexpectedly. Check the terminal for details.")

    def infer(self, reference, transcript, text, seed):
        from config import OUTPUT_DIR
        with tempfile.TemporaryDirectory(prefix="f5_", dir=OUTPUT_DIR) as directory:
            output_path = os.path.join(directory, "audio.npz")
            request = dict(reference=reference, transcript=transcript, text=text,
                           seed=seed, output_path=output_path)
            try:
                self.process.stdin.write(json.dumps(request) + "\n")
                self.process.stdin.flush()
            except (BrokenPipeError, OSError) as exc:
                raise RuntimeError("F5-TTS worker exited unexpectedly.") from exc
            result = self._read_result()
            with np.load(output_path) as data:
                return np.asarray(data["wav"], dtype=np.float32), int(result["sample_rate"])

    def close(self):
        atexit.unregister(self.close)
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        if self.process.stdin:
            self.process.stdin.close()
        if self.process.stdout:
            self.process.stdout.close()


def run_worker(device):
    def respond(**result):
        print(_PREFIX + json.dumps(result), flush=True)

    try:
        with contextlib.redirect_stdout(sys.stderr):
            from f5_tts.api import F5TTS
            model = F5TTS(device=device)
    except Exception as exc:
        respond(ok=False, error=f"Could not load F5-TTS: {exc}. See README for F5 setup.")
        return
    respond(ok=True)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            with contextlib.redirect_stdout(sys.stderr):
                wav, sample_rate, _ = model.infer(
                    ref_file=request["reference"], ref_text=request["transcript"],
                    gen_text=request["text"], seed=request["seed"], progress=None,
                )
            if wav is None or not len(wav):
                raise ValueError("F5-TTS returned no audio.")
            np.savez(request["output_path"], wav=np.asarray(wav, dtype=np.float32))
            respond(ok=True, sample_rate=int(sample_rate))
        except Exception as exc:
            respond(ok=False, error=f"F5-TTS generation failed: {exc}")


if __name__ == "__main__":
    run_worker(sys.argv[1])