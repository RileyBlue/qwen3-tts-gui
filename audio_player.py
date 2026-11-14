"""Minimal WAV playback: pygame if available, else a system player."""

import shutil
import subprocess
import sys

_proc = None


def _pygame():
    try:
        import pygame
        if not pygame.mixer.get_init():
            pygame.mixer.init()
        return pygame
    except Exception:
        return None


def stop():
    global _proc
    pg = _pygame()
    if pg:
        pg.mixer.music.stop()
    if _proc and _proc.poll() is None:
        _proc.terminate()
    _proc = None


def play(path):
    """Start playing `path` without blocking. Raises RuntimeError if nothing can play it."""
    global _proc
    stop()
    pg = _pygame()
    if pg:
        pg.mixer.music.load(path)
        pg.mixer.music.play()
        return
    for cmd in (["pw-play"], ["paplay"], ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"]):
        if shutil.which(cmd[0]):
            _proc = subprocess.Popen(cmd + [path])
            return
    raise RuntimeError("No audio player found (install pygame, or paplay/pw-play/ffplay).")


def is_playing():
    """True while audio started by play() is still audible. Read-only: never starts or stops anything.

    pygame is only asked if it is already imported and its mixer already initialised, so asking never
    opens an audio device. The system-player fallback is a subprocess that is alive while it plays.
    """
    pg = sys.modules.get("pygame")
    try:
        if pg is not None and pg.mixer.get_init() and pg.mixer.music.get_busy():
            return True
    except Exception:
        pass
    return bool(_proc and _proc.poll() is None)
