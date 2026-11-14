"""What went wrong in a background task, in words a person can act on.

`Failure` keeps the original exception's facts (type, message, traceback, errno) so nothing about the
cause is lost. `classify` names a cause only when it can be identified reliably, from the exception
type or from text the exception itself contains; anything else stays an honest "unknown". `details_text`
builds the technical text shown under "Show details" and copied to the clipboard. It is sanitized:
home folders, the app folder, credentials and email addresses are removed. The raw traceback still goes
to the console (stderr) for local diagnosis.

Nothing here changes how models load or run.
"""

import errno as _errno
import os
import platform
import re
import sys
import traceback
from dataclasses import dataclass
from typing import Optional

UNKNOWN_SUMMARY = "Something went wrong while creating the sample."
UNKNOWN_ACTION = "Your description is unchanged. You can try again, or open Show details to see what happened."


@dataclass(frozen=True)
class Failure:
    type_name: str            # "module.ClassName", e.g. "builtins.ImportError"
    message: str              # str(exception), as raised
    trace: str                # formatted traceback, as raised (not sanitized)
    errno: Optional[int] = None

    @property
    def short_type(self):
        return self.type_name.rsplit(".", 1)[-1]

    @classmethod
    def from_exception(cls, exc):
        kind = type(exc)
        return cls(
            type_name=f"{kind.__module__}.{kind.__qualname__}",
            message=str(exc) or kind.__name__,
            trace="".join(traceback.format_exception(kind, exc, exc.__traceback__)),
            errno=getattr(exc, "errno", None) if isinstance(exc, OSError) else None,
        )

    @classmethod
    def from_message(cls, message):
        """A failure known only by its text (nothing else was captured)."""
        return cls(type_name="", message=message, trace="")


@dataclass(frozen=True)
class Explanation:
    kind: str       # flash | memory | disk | network | package
    summary: str    # what failed, one sentence
    action: str     # what the person can do next (a suggestion, never a promised fix)
    short: str      # a few words for the footer
    settings: bool  # the Settings page is where the person can act


_NETWORK_TYPES = ("ConnectionError", "ConnectTimeout", "ReadTimeout", "Timeout", "OfflineModeIsEnabled",
                  "LocalEntryNotFoundError", "ConnectError", "NameResolutionError", "MaxRetryError")


def classify(failure):
    """Return an Explanation for a reliably identifiable cause, else None (unknown)."""
    kind, text = failure.short_type, failure.message
    lowered = text.lower()
    if kind.endswith("ImportError") or kind == "ModuleNotFoundError":
        if re.search(r"flash[_ ]?attn|flashattention", text, re.I):
            return Explanation(
                "flash", "The selected Attention option needs FlashAttention, which isn't installed on this computer.",
                "Choose a different Attention option in Settings, then try again.",
                "FlashAttention isn't installed", True)
        if "package is not installed" in lowered:
            return Explanation(
                "package", "A required package isn't installed.",
                "Show details has the install command. Restart the app after installing it.",
                "a required package is missing", False)
    if kind in ("OutOfMemoryError", "MemoryError") or "out of memory" in lowered:
        return Explanation(
            "memory", "There isn't enough memory to create the sample.",
            "Closing other apps may free some memory. Then try again.", "not enough memory", False)
    if failure.errno == _errno.ENOSPC:
        return Explanation(
            "disk", "There isn't enough free disk space.",
            "Free some space, then try again.", "not enough disk space", False)
    if kind in _NETWORK_TYPES:
        return Explanation(
            "network", "Couldn't reach the model server. The voice model may need to be downloaded or checked online.",
            "Check your internet connection, then try again.", "can't reach the model server", False)
    return None


# ---------------------------------------------------------------------------------- sanitizing
_SECRETS = (
    (re.compile(r"\bhf_[A-Za-z0-9]{10,}"), "<token>"),
    (re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._\-]+"), "Bearer <token>"),
    (re.compile(r"(?i)\b(token|api[_-]?key|secret|password|authorization)(\s*[=:]\s*)[^\s,;'\"]+"), r"\1\2<redacted>"),
    (re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}"), "<token>"),
    (re.compile(r"[\w.+\-]+@[\w\-]+\.[\w.\-]+"), "<email>"),
)


def sanitize(text, home=None, app_dir=None):
    """Remove private paths, credentials and emails from text meant to be shown or shared."""
    home = home if home is not None else os.path.expanduser("~")
    app_dir = app_dir if app_dir is not None else os.path.dirname(os.path.abspath(__file__))
    for path, token in sorted(((app_dir, "<app>"), (home, "~")), key=lambda p: -len(p[0] or "")):
        if path and path not in ("/", "~"):
            text = text.replace(path, token)
    text = re.sub(r"[^\s\"']*/site-packages/", "<site-packages>/", text)
    text = re.sub(r"(/Users|/home)/[^/\s\"']+", r"\1/<user>", text)
    text = re.sub(r"(?i)([A-Z]:\\Users\\)[^\\\s\"']+", r"\1<user>", text)
    for pattern, replacement in _SECRETS:
        text = pattern.sub(replacement, text)
    return text


def details_text(failure, task="Create Voice sample", context=None, home=None, app_dir=None):
    """Technical details for display and copying, sanitized. `context` is {label: value} of settings."""
    lines = ["Qwen3-TTS Studio technical details", f"Task: {task}"]
    if failure.type_name:
        lines.append(f"Error type: {failure.type_name}")
    lines.append(f"Message: {failure.message}")
    for label, value in (context or {}).items():
        lines.append(f"{label}: {value}")
    lines.append(f"Platform: {platform.platform()} · Python {platform.python_version()}")
    if failure.trace:
        lines += ["", failure.trace.rstrip()]
    return sanitize("\n".join(lines), home=home, app_dir=app_dir)


def log_failure(failure):
    """Full, unsanitized traceback to the console for local diagnosis."""
    print(failure.trace or f"Error: {failure.message}", file=sys.stderr)
