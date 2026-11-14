"""Minimal WAV playback: pygame if available, else a system player.

pygame (its mixer.music stream) also gives pause/resume, a position and seeking; the system-player
fallback (pw-play, paplay, ffplay) can only start and stop, so `can_control()` says which one is in
use and callers hide what it can't do. Nothing here knows durations: the caller owns the audio and
knows its length.
"""

import importlib.util
import shutil
import subprocess
import sys

_proc = None
_path = None            # file started by play(), while it is playing or paused
_paused = False
_offset = 0.0           # seconds: position = _offset + pygame's own play-time clock
_backend = None         # "pygame" | "process" | None


def _pygame():
    try:
        import pygame
        if not pygame.mixer.get_init():
            pygame.mixer.init()
        return pygame
    except Exception:
        return None


def _music():
    pg = sys.modules.get("pygame")
    try:
        return pg.mixer.music if pg is not None and pg.mixer.get_init() else None
    except Exception:
        return None


def stop():
    global _proc, _path, _paused, _offset, _backend
    pg = _pygame()
    if pg:
        pg.mixer.music.stop()
    if _proc and _proc.poll() is None:
        _proc.terminate()
    _proc = None
    _path, _paused, _offset, _backend = None, False, 0.0, None


def play(path, start=0.0):
    """Start playing `path` (from `start` seconds where the backend can seek) without blocking.
    Raises RuntimeError if nothing can play it."""
    global _proc, _path, _offset, _backend
    stop()
    pg = _pygame()
    if pg:
        pg.mixer.music.load(path)
        pg.mixer.music.play()
        _path, _backend, _offset = path, "pygame", 0.0
        if start > 0:
            seek(start)
        return
    for cmd in (["pw-play"], ["paplay"], ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"]):
        if shutil.which(cmd[0]):
            _proc = subprocess.Popen(cmd + [path])
            _path, _backend = path, "process"
            return
    raise RuntimeError("No audio player found (install pygame, or paplay/pw-play/ffplay).")


def is_playing():
    """True while audio started by play() is still audible (not paused, not ended). Read-only: never
    starts or stops anything.

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


def state():
    """'playing', 'paused' or 'stopped' (never started, stopped, or played to the end)."""
    if _paused and _backend == "pygame":
        return "paused"
    return "playing" if is_playing() else "stopped"


def current_path():
    """The file that is playing or paused, else None."""
    return _path if state() != "stopped" else None


def can_control():
    """True when pause/resume, position and seek are available (pygame). Does not open an audio device."""
    if _backend:
        return _backend == "pygame"
    return importlib.util.find_spec("pygame") is not None


def pause():
    global _paused
    music = _music()
    if _backend == "pygame" and music is not None and state() == "playing":
        music.pause()
        _paused = True
        return True
    return False


def resume():
    global _paused
    music = _music()
    if _backend == "pygame" and music is not None and _paused:
        music.unpause()
        _paused = False
        return True
    return False


def position():
    """Seconds into the current audio, or None when it isn't playing/paused or can't be known."""
    music = _music()
    if _backend != "pygame" or music is None or state() == "stopped":
        return None
    played = music.get_pos()
    return None if played < 0 else max(0.0, _offset + played / 1000.0)


def seek(seconds):
    """Jump to `seconds` in the audio that is playing or paused. Returns False if it can't."""
    global _offset
    music = _music()
    if _backend != "pygame" or music is None or state() == "stopped":
        return False
    try:
        music.set_pos(max(0.0, float(seconds)))
        played = max(0, music.get_pos())
    except Exception:
        return False
    _offset = max(0.0, float(seconds)) - played / 1000.0   # pygame's clock keeps counting through a seek
    return True
