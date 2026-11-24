"""UI-independent project and settings orchestration for Drama Studio."""

import hashlib
import json
import os
from pathlib import Path

import minimax_drama as drama
import voice_library

BASE_DIR = Path(__file__).resolve().parent
SETTINGS_FILE = BASE_DIR / "drama_settings.json"
PROVIDERS = {
    "OpenRouter": ("https://openrouter.ai/api/v1", "openai/gpt-4o-mini", "OPENROUTER_API_KEY"),
    "OpenAI": ("https://api.openai.com/v1", "gpt-4o-mini", "OPENAI_API_KEY"),
    "Custom / local": ("http://127.0.0.1:11434/v1", "qwen2.5:32b-instruct", ""),
}
DEFAULT_SETTINGS = {"comfy_url": "http://127.0.0.1:8188", "provider": "OpenRouter",
                    "llm_base_url": PROVIDERS["OpenRouter"][0], "llm_model": PROVIDERS["OpenRouter"][1]}


def load_settings():
    settings = dict(DEFAULT_SETTINGS)
    if SETTINGS_FILE.exists():
        try:
            saved = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
            settings.update({key: saved[key] for key in DEFAULT_SETTINGS if key in saved})
        except (OSError, ValueError, TypeError):
            pass
    return settings


def save_settings(settings):
    """Persist connection preferences only; never serialize an API key."""
    safe = {key: settings.get(key, DEFAULT_SETTINGS[key]) for key in DEFAULT_SETTINGS}
    SETTINGS_FILE.write_text(json.dumps(safe, indent=2), encoding="utf-8")
    return safe


def environment_key(provider):
    entry = PROVIDERS.get(provider)
    return os.environ.get(entry[2], "") if entry and entry[2] else ""


def voice_reference_path(voice_name, root=None):
    """Return the path to a saved voice's reference audio file, or None if missing."""
    if not voice_name:
        return None
    folder = Path(voice_library.voice_dir(voice_name, root))
    ref_name = "reference.wav"
    cfg_path = folder / "voice.json"
    if cfg_path.is_file():
        try:
            ref_name = json.loads(cfg_path.read_text(encoding="utf-8")).get("ref_audio") or ref_name
        except (OSError, ValueError):
            pass
    ref_path = folder / ref_name
    return str(ref_path) if ref_path.is_file() else None


def _saved_voice_description(voice_name, root=None):
    if not voice_name:
        return ""
    cfg_path = Path(voice_library.voice_dir(voice_name, root)) / "voice.json"
    if not cfg_path.is_file():
        return ""
    try:
        return json.loads(cfg_path.read_text(encoding="utf-8")).get("description", "")
    except (OSError, ValueError):
        return ""


class DramaStudio:
    @staticmethod
    def list_projects():
        if not drama.PROJECTS_DIR.is_dir():
            return []
        return sorted(entry.name for entry in drama.PROJECTS_DIR.iterdir()
                      if entry.is_dir() and not entry.name.startswith("."))

    @staticmethod
    def project_dir(project):
        path = drama.project_dir(project)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def references_dir(project):
        path = drama.project_references_dir(project)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def characters(self, project):
        return drama.load_registry(self.references_dir(project))

    def save_character(self, project, character):
        refs_dir = self.references_dir(project)
        registry = drama.load_registry(refs_dir)
        item = dict(character)
        item["id"] = drama.slug(item.get("id") or item.get("name"))
        if not item.get("name", "").strip():
            raise ValueError("Character name is required.")
        if item.get("gender") not in ("male", "female"):
            raise ValueError("Character gender must be male or female.")
        existing = registry.get(item["id"], {})
        item.setdefault("needs_reference", existing.get("needs_reference", True))
        item.setdefault("appearance", "")
        item.setdefault("ref_tags", "")
        item.setdefault("voice_name", "")
        item.setdefault("reference", "")
        static_voice = item.get("voice") or existing.get("voice") or _saved_voice_description(item["voice_name"])
        item["voice"] = static_voice
        registry[item["id"]] = item
        drama.save_json(refs_dir / "characters.json", registry)
        return item

    def delete_character(self, project, character_id):
        refs_dir = self.references_dir(project)
        registry = drama.load_registry(refs_dir)
        registry.pop(character_id, None)
        drama.save_json(refs_dir / "characters.json", registry)

    def generate_plan(self, source, project, settings, api_key):
        if not source.strip():
            raise ValueError("Enter prose or a prompt before generating a script.")
        if not settings.get("llm_base_url") or not settings.get("llm_model"):
            raise ValueError("Configure the LLM URL and model in Connection settings.")
        if not api_key and settings.get("provider") != "Custom / local":
            raise ValueError("Enter an API key or set the provider's environment variable.")
        project_dir = self.project_dir(project)
        refs_dir = self.references_dir(project)
        registry = drama.load_registry(refs_dir)
        raw = drama.call_llm(
            drama.build_system_prompt(registry), source,
            base_url=settings["llm_base_url"], api_key=api_key or "local",
            model=settings["llm_model"])
        (project_dir / "llm_raw.txt").write_text(raw, encoding="utf-8")
        drama.check_failsafe_text(raw)
        plan = drama.validate_plan(drama.extract_json(raw), source)
        plan["source_hash"] = hashlib.sha256(source.encode()).hexdigest()
        drama.save_json(project_dir / "plan.json", plan)
        drama.merge_registry(registry, plan, refs_dir)
        return plan

    def save_plan(self, plan, source, project):
        refs_dir = self.references_dir(project)
        plan = drama.validate_plan(plan, source)
        plan["source_hash"] = hashlib.sha256(source.encode()).hexdigest()
        drama.save_json(self.project_dir(project) / "plan.json", plan)
        drama.merge_registry(drama.load_registry(refs_dir), plan, refs_dir)
        return plan

    def load_plan(self, project):
        return drama.load_json(self.project_dir(project) / "plan.json")

    def generate_images(self, plan, project, settings):
        refs_dir = self.references_dir(project)
        comfy = drama.Comfy(settings["comfy_url"])
        paths, generated = drama.ensure_references(
            comfy, plan, refs_dir=refs_dir, registry=drama.load_registry(refs_dir))
        return {key: str(value) for key, value in paths.items()}, generated

    def render_video(self, plan, project, settings, seed=None, lightning=False, force=False):
        project_dir = self.project_dir(project)
        refs_dir = self.references_dir(project)
        comfy = drama.Comfy(settings["comfy_url"])
        references, _ = drama.ensure_references(
            comfy, plan, refs_dir=refs_dir, registry=drama.load_registry(refs_dir))
        uploaded = {key: comfy.upload_image(path) for key, path in references.items()}
        clips = drama.render_segments(comfy, plan, uploaded, drama.slug(project),
                                      project_dir / "segments", seed, lightning, force)
        final = project_dir / f"{drama.slug(project)}_final.mp4"
        drama.concat_segments(clips, final)
        return str(final)
