"""On-disk library of designed voices.

Each voice lives in voices/<slug>/ :
    voice.json        metadata (description, seed, sample text, file names, usage notes)
    reference.wav     the approved VoiceDesign sample
    clone_prompt.npz  prebuilt Base-model clone prompt (speaker embedding + ref codes + transcript)
    emotions/         cached per-emotion clone prompts (created on demand)
"""

import contextlib
import json
import os
import re
import secrets
import shutil
import threading
import time
from datetime import datetime
import random

import numpy as np

from audio_utils import load_voice_clone_prompt, save_voice_clone_prompt, write_wav, load_wav, AudioReadError

VOICES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "voices")

SAMPLE_TEXT = (
    "Are you certain the quick, brown fox jumped over the lazy, yawning dog? "
    "Honestly, I couldn't believe my own eyes! The majestic creature moved with "
    "such vivid, explosive power. Breathing deeply, we just watched the bizarre "
    "scene unfold, marveling at the quiet, azure sky above."
)

USAGE_NOTES = {
    "voice_clone": (
        "Pass clone_prompt.npz as voice_clone_prompt to Qwen3TTSModel.generate_voice_clone() on the "
        "Base model, or use reference.wav + ref_text with create_voice_clone_prompt()."
    ),
    "voice_design": (
        "description is the VoiceDesign instruct. Seed does not guarantee the same voice for new text; "
        "use the clone prompt for a stable identity."
    ),
}


def slugify(name):
    slug = re.sub(r"[^\w\- ]+", "", name, flags=re.UNICODE).strip().replace(" ", "_")
    return slug[:60]


def voice_dir(name, root=None):
    return os.path.join(root or VOICES_DIR, slugify(name))


def list_voices(root=None):
    """Return sorted voice names that have a valid voice.json."""
    root = root or VOICES_DIR
    names = []
    if os.path.isdir(root):
        for entry in os.listdir(root):
            cfg = os.path.join(root, entry, "voice.json")
            if os.path.isfile(cfg):
                try:
                    with open(cfg, encoding="utf-8") as f:
                        names.append(json.load(f)["name"])
                except (OSError, ValueError, KeyError):
                    continue
    return sorted(names, key=str.lower)


def voice_exists(name, root=None):
    return os.path.isfile(os.path.join(voice_dir(name, root), "voice.json"))


class VoiceSaveError(RuntimeError):
    """A save or replace failed. The message says what state the existing voice is in."""


_PENDING_DIR = ".pending"        # slugify() strips dots, so no voice can ever be named like this
_save_lock = threading.RLock()


LOCK_TIMEOUT = 10.0   # seconds a save waits for another Qwen3-TTS window that is saving or recovering


def _try_lock(handle):
    """Take an exclusive, non-blocking lock on an open file. True when held. Unsupported platforms: True (no protection)."""
    try:
        import fcntl
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except OSError:
            return False
    except ImportError:
        pass
    try:
        import msvcrt
        handle.seek(0)
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            return True
        except OSError:
            return False
    except ImportError:
        return True


def _unlock(handle):
    try:
        import fcntl
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return
    except ImportError:
        pass
    except OSError:
        return
    try:
        import msvcrt
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    except (ImportError, OSError):
        pass


@contextlib.contextmanager
def _root_lock(root, timeout):
    """Cross-process lock on the voices folder, held while a save or a recovery rewrites it. Yields False when it
    could not be taken within `timeout` seconds. The OS drops it if the process dies, so it can never go stale."""
    try:
        os.makedirs(root, exist_ok=True)
        handle = open(os.path.join(root, ".voices.lock"), "a+")
    except OSError as exc:
        raise VoiceSaveError(f"Couldn't save the voice ({exc}). Nothing was changed.") from exc
    held = False
    try:
        deadline = time.monotonic() + timeout
        while True:
            held = _try_lock(handle)
            if held or time.monotonic() >= deadline:
                break
            time.sleep(0.05)
        yield held
    finally:
        if held:
            _unlock(handle)
        handle.close()


def _pending_root(root=None):
    return os.path.join(root or VOICES_DIR, _PENDING_DIR)


def _write_text_durably(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass


def _fsync_path(path):
    """Best-effort flush of a file or folder to disk (some platforms cannot open folders)."""
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def recover_interrupted_saves(root=None):
    """Public entry point (app start): skipped, returning [], while another window is saving."""
    root = root or VOICES_DIR
    if not os.path.isdir(_pending_root(root)):
        return []
    try:
        with _save_lock, _root_lock(root, 0) as held:
            return _recover_locked(root) if held else []
    except VoiceSaveError:
        return []   # a read-only folder has nothing to recover, and start-up must not fail on it


def _recover_locked(root=None):
    """Finish or undo a replace that was interrupted (crash) between its two renames.

    A previous version parked in .pending/<slug>.old-* is put back when the voice folder is missing,
    and discarded when the new version is already in place. Half-written .new-* folders and finished
    .done-* leftovers (the replace succeeded but deleting the old copy failed) are removed; a .done-*
    folder is never restored, so a voice the user deleted afterwards stays deleted.
    Returns the slugs that were restored."""
    root = root or VOICES_DIR
    pending = _pending_root(root)
    restored = []
    if not os.path.isdir(pending):
        return restored
    with _save_lock:
        for entry in sorted(os.listdir(pending)):
            full = os.path.join(pending, entry)
            if ".done-" in entry or ".new-" in entry:
                shutil.rmtree(full, ignore_errors=True)
            elif ".old-" in entry:
                slug = entry.rsplit(".old-", 1)[0]
                final = os.path.join(root, slug)
                if os.path.lexists(final):
                    shutil.rmtree(full, ignore_errors=True)
                else:
                    try:
                        os.rename(full, final)
                        restored.append(slug)
                    except OSError:
                        pass   # left in place; nothing is deleted when it cannot be restored
        try:
            os.rmdir(pending)
        except OSError:
            pass
    return restored


def describe_voice_assets(name, root=None):
    """Plain-language list of what replacing this voice discards, for the confirmation dialog."""
    path = voice_dir(name, root)
    items = []
    if os.path.isfile(os.path.join(path, "reference.wav")):
        items.append("its recording and clone prompt")
    emotions = os.path.join(path, "emotions")
    if os.path.isdir(emotions):
        count = len([e for e in os.listdir(emotions) if e.endswith(".npz")])
        if count:
            items.append(f"{count} cached emotion sample{'s' if count != 1 else ''}")
    try:
        with open(os.path.join(path, "voice.json"), encoding="utf-8") as f:
            config = json.load(f)
    except (OSError, ValueError):
        config = {}
    if config.get("f5_ref_audio") or config.get("f5_ref_text"):
        items.append("its F5 reference recording and transcript")
    if config.get("csm_ref_text"):
        items.append("its CSM/Dia transcript")
    return items


def save_voice(name, wav, sr, prompt, description, language, seed,
               model_ids=None, root=None, overwrite=False, sample_text=None):
    """Write a voice folder and return its config dict. `sample_text` is the text spoken in `wav`
    (None: the standard audition passage; "": the recording has no known transcript, stored as empty).
    It is stored as sample_text and ref_text.

    The new voice is built in a staging folder first and only then moved into place, so a failed write
    never touches an existing voice (files are flushed before the swap; durability across power loss is
    best effort and platform dependent). Replacing renames the old folder aside, moves the new one in and
    deletes the old one last; if the move fails the old folder is put back. Assets bound to the old
    recording (emotion cache, F5 reference, CSM transcript) are not carried over: they would describe
    a different speaker."""
    sample_text = SAMPLE_TEXT if sample_text is None else sample_text   # "" is a real value: no transcript
    slug = slugify(name)
    if not slug:
        raise ValueError("Voice name must contain letters or digits.")
    path = voice_dir(name, root)
    with _save_lock, _root_lock(root or VOICES_DIR, LOCK_TIMEOUT) as held:
        if not held:
            raise VoiceSaveError("Another Qwen3-TTS window is saving a voice right now. "
                                 "Try again in a moment. Nothing was changed.")
        _recover_locked(root)
        exists = os.path.lexists(path)
        if exists and not overwrite:
            raise FileExistsError(f"A voice named '{name}' already exists.")
        if exists and (os.path.islink(path) or not os.path.isfile(os.path.join(path, "voice.json"))):
            raise FileExistsError(f"'{os.path.basename(path)}' exists in the voices folder but is not a saved voice.")
        pending = _pending_root(root)
        token = secrets.token_hex(6)
        stage = os.path.join(pending, f"{slug}.new-{token}")
        backup = os.path.join(pending, f"{slug}.old-{token}")
        kept = "The existing voice is unchanged." if exists else "Nothing was saved."
        try:
            os.makedirs(pending, exist_ok=True)
            os.mkdir(stage)
            write_wav(os.path.join(stage, "reference.wav"), np.asarray(wav), sr)
            save_voice_clone_prompt(os.path.join(stage, "clone_prompt.npz"), prompt)
            config = {
                "schema": 1,
                "name": name.strip(),
                "created": datetime.now().isoformat(timespec="seconds"),
                "language": language,
                "description": description,
                "seed": int(seed),
                "sample_text": sample_text,
                "ref_text": sample_text,
                "ref_audio": "reference.wav",
                "clone_prompt": "clone_prompt.npz",
                "sample_rate": int(sr),
                "models": model_ids or {},
                "usage": USAGE_NOTES,
            }
            _write_text_durably(os.path.join(stage, "voice.json"), json.dumps(config, indent=2, ensure_ascii=False))
            for member in ("reference.wav", "clone_prompt.npz"):
                _fsync_path(os.path.join(stage, member))
            _fsync_path(stage)
        except BaseException as exc:
            shutil.rmtree(stage, ignore_errors=True)
            if isinstance(exc, OSError):
                raise VoiceSaveError(f"Couldn't save the voice ({exc}). {kept}") from exc
            raise

        moved_old = False
        try:
            if os.path.lexists(path):
                if not overwrite:   # appeared while the files were being written
                    raise FileExistsError(f"A voice named '{name}' already exists.")
                os.rename(path, backup)
                moved_old = True
            os.rename(stage, path)
        except BaseException as exc:
            shutil.rmtree(stage, ignore_errors=True)
            if not moved_old:
                if isinstance(exc, FileExistsError) or not isinstance(exc, Exception):
                    raise
                raise VoiceSaveError(f"Couldn't save the voice ({exc}). Nothing was changed.") from exc
            try:
                os.rename(backup, path)
            except OSError:
                raise VoiceSaveError(
                    f"Couldn't save the voice ({exc}), and the previous version could not be put back. "
                    f"It is kept at {backup}.") from exc
            if not isinstance(exc, Exception):
                raise
            raise VoiceSaveError(f"Couldn't save the voice ({exc}). The previous version is unchanged.") from exc
        if moved_old:
            # The save has succeeded. Rename the old copy so that, if deleting it fails, recovery can tell it
            # is finished business and never restores it over a voice the user later deletes or renames.
            leftover = backup.replace(".old-", ".done-")
            try:
                os.rename(backup, leftover)
            except OSError:
                leftover = backup
            shutil.rmtree(leftover, ignore_errors=True)
        try:
            os.rmdir(pending)
        except OSError:
            pass
        return config


def load_voice(name, root=None):
    """Return (config, clone_prompt, voice_directory) for a saved voice."""
    path = voice_dir(name, root)
    with open(os.path.join(path, "voice.json"), encoding="utf-8") as f:
        config = json.load(f)
    prompt = load_voice_clone_prompt(os.path.join(path, config["clone_prompt"]))
    return config, prompt, path


def set_csm_reference_text(name, text, root=None):
    path = os.path.join(voice_dir(name, root), "voice.json")
    with open(path, encoding="utf-8") as voice_file:
        config = json.load(voice_file)
    config["csm_ref_text"] = text.strip()
    with open(path, "w", encoding="utf-8") as voice_file:
        json.dump(config, voice_file, indent=2, ensure_ascii=False)


def set_f5_reference(name, source_path, text, root=None):
    text = text.strip()
    if not text:
        raise ValueError("Enter the transcript for the F5 reference recording.")
    try:
        wav, sample_rate = load_wav(source_path)
    except AudioReadError as exc:   # load_wav converts every decoder failure into this
        raise ValueError(f"F5 reference: {exc}") from exc
    if not len(wav) or len(wav) / sample_rate > 12:
        raise ValueError("F5 reference audio must be nonempty and no longer than 12 seconds.")
    path = voice_dir(name, root)
    config_path = os.path.join(path, "voice.json")
    with open(config_path, encoding="utf-8") as voice_file:
        config = json.load(voice_file)
    write_wav(os.path.join(path, "f5_reference.wav"), wav, sample_rate)
    config["f5_ref_audio"] = "f5_reference.wav"
    config["f5_ref_text"] = text
    with open(config_path, "w", encoding="utf-8") as voice_file:
        json.dump(config, voice_file, indent=2, ensure_ascii=False)


def delete_voice(name, root=None):
    path = voice_dir(name, root)
    if os.path.isfile(os.path.join(path, "voice.json")):
        shutil.rmtree(path)


# ------------------------------------------------------------------ random name generation
_MALE_FIRST = [
    "James", "Robert", "John", "Michael", "David", "William", "Richard", "Joseph",
    "Thomas", "Charles", "Christopher", "Daniel", "Matthew", "Anthony", "Mark",
    "Donald", "Steven", "Paul", "Andrew", "Joshua", "Kevin", "Brian", "George",
    "Edward", "Ronald", "Timothy", "Jason", "Jeffrey", "Ryan", "Jacob", "Gary",
    "Nicholas", "Eric", "Jonathan", "Stephen", "Larry", "Justin", "Scott", "Brandon",
    "Benjamin", "Samuel", "Gregory", "Frank", "Alexander", "Raymond", "Jack",
    "Patrick", "Dennis", "Jerry", "Tyler", "Aaron", "Jose", "Adam", "Henry",
    "Nathan", "Douglas", "Zachary", "Peter", "Kyle", "Walter", "Ethan", "Jeremy",
    "Harold", "Keith", "Christian", "Roger", "Noah", "Gerald", "Carl", "Keith",
]
_FEMALE_FIRST = [
    "Mary", "Patricia", "Jennifer", "Linda", "Elizabeth", "Barbara", "Susan",
    "Jessica", "Sarah", "Karen", "Lisa", "Nancy", "Betty", "Sandra", "Margaret",
    "Ashley", "Kimberly", "Emily", "Donna", "Michelle", "Dorothy", "Carol",
    "Amanda", "Melissa", "Deborah", "Stephanie", "Rebecca", "Sharon", "Laura",
    "Cynthia", "Kathleen", "Amy", "Angela", "Helen", "Anna", "Brenda", "Pamela",
    "Nicole", "Emma", "Samantha", "Katherine", "Christine", "Janet", "Debra",
    "Rachel", "Catherine", "Heather", "Diane", "Julie", "Joyce", "Victoria",
    "Kelly", "Christina", "Joan", "Lauren", "Martha", "Victoria", "Alicia",
    "Julie", "Cheryl", "Megan", "Andrea", "Hannah", "Katie", "Natalie",
]
_LAST = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
    "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson",
    "Thomas", "Taylor", "Moore", "Jackson", "Martin", "Lee", "Perez", "Thompson",
    "White", "Harris", "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson", "Walker",
    "Young", "Allen", "King", "Wright", "Scott", "Torres", "Nguyen", "Flores",
    "Edwards", "Collins", "Stewart", "Sanchez", "Morris", "Rogers", "Reed",
    "Cook", "Morgan", "Bell", "Murphy", "Bailey", "Rivera", "Cooper", "Richardson",
    "Cox", "Howard", "Ward", "Patterson", "Hughes", "Flores", "Washington",
    "Butler", "Simmons", "Foster", "Gonzales", "Bryant", "Alexander", "Russell",
]


def random_voice_name(gender="random"):
    """Generate a random voice name.

    gender: "male", "female", or "random".
    Returns a full name like "James Smith".
    """
    if gender == "male":
        first = random.choice(_MALE_FIRST)
    elif gender == "female":
        first = random.choice(_FEMALE_FIRST)
    else:
        first = random.choice(_MALE_FIRST + _FEMALE_FIRST)
    last = random.choice(_LAST)
    return f"{first} {last}"


def unique_voice_name(base_name, root=None):
    """Return a unique variant of base_name by appending a number if needed."""
    root = root or VOICES_DIR
    if not voice_exists(base_name, root):
        return base_name
    i = 2
    while voice_exists(f"{base_name} {i}", root):
        i += 1
    return f"{base_name} {i}"


def list_voice_details(root=None):
    """Return list of voice detail dicts for all saved voices.

    Each dict: name, slug, description, language, seed, created, path.
    """
    root = root or VOICES_DIR
    details = []
    if not os.path.isdir(root):
        return details
    for entry in sorted(os.listdir(root)):
        cfg_path = os.path.join(root, entry, "voice.json")
        if not os.path.isfile(cfg_path):
            continue
        try:
            with open(cfg_path, encoding="utf-8") as f:
                cfg = json.load(f)
            details.append({
                "name": cfg.get("name", entry),
                "slug": entry,
                "description": cfg.get("description", ""),
                "language": cfg.get("language", ""),
                "seed": cfg.get("seed", 0),
                "created": cfg.get("created", ""),
                "path": os.path.join(root, entry),
            })
        except (OSError, ValueError, KeyError):
            continue
    return details


def rename_voice(old_name, new_name, root=None):
    """Rename a saved voice directory and update voice.json name field."""
    root = root or VOICES_DIR
    old_slug = slugify(old_name)
    new_slug = slugify(new_name)
    if not new_slug:
        raise ValueError("New voice name must contain letters or digits.")
    old_path = os.path.join(root, old_slug)
    new_path = os.path.join(root, new_slug)
    if not os.path.isdir(old_path):
        raise FileNotFoundError(f"Voice '{old_name}' not found.")
    if os.path.exists(new_path) and old_slug != new_slug:
        raise FileExistsError(f"A voice named '{new_name}' already exists.")
    if old_slug != new_slug:
        os.rename(old_path, new_path)
    cfg_path = os.path.join(new_path, "voice.json")
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["name"] = new_name.strip()
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    return new_name
