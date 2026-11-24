"""Non-GUI tests for Drama Studio orchestration and minimax_drama integration."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import drama_studio
import minimax_drama as drama


class DramaStudioTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_workflow_files_exist(self):
        self.assertTrue(drama.VIDEO_WORKFLOW.is_file())
        self.assertTrue(drama.IMAGE_WORKFLOW.is_file())
        self.assertTrue(drama.TEMPLATE_FILE.is_file())

    def test_save_settings_never_writes_api_key(self):
        settings_path = self.root / "drama_settings.json"
        with patch.object(drama_studio, "SETTINGS_FILE", settings_path):
            saved = drama_studio.save_settings({
                "comfy_url": "http://remote-host:8188",
                "provider": "OpenRouter",
                "llm_base_url": "https://openrouter.ai/api/v1",
                "llm_model": "openai/gpt-4o-mini",
                "api_key": "sk-secret-never-save",
            })
            raw = settings_path.read_text(encoding="utf-8")
            self.assertNotIn("sk-secret-never-save", raw)
            self.assertEqual(saved["comfy_url"], "http://remote-host:8188")

    def test_cast_persistence_and_assigned_reference(self):
        projects = self.root / "projects"
        assigned = self.root / "custom_molly.png"
        assigned.write_bytes(b"png")
        with patch.object(drama, "PROJECTS_DIR", projects):
            studio = drama_studio.DramaStudio()
            saved = studio.save_character("Pilot", {
                "name": "Molly Blunt", "gender": "female", "voice_name": "MollyVoice",
                "voice": "dry deadpan female voice", "reference": str(assigned),
                "appearance": "dark ponytail", "ref_tags": "messy dark ponytail",
            })
            refs_dir = drama.project_references_dir("pilot")
            self.assertEqual(saved["id"], "molly_blunt")
            self.assertEqual(drama.find_reference("molly_blunt", saved, refs_dir=refs_dir), assigned)
            self.assertEqual(studio.characters("Pilot")["molly_blunt"]["voice_name"], "MollyVoice")
            self.assertTrue((refs_dir / "characters.json").is_file())
            self.assertEqual(studio.list_projects(), ["pilot"])

    def test_voice_reference_path_and_static_voice_description(self):
        voices = self.root / "voices"
        vdir = voices / "MollyVoice"
        vdir.mkdir(parents=True)
        ref_wav = vdir / "reference.wav"
        ref_wav.write_bytes(b"RIFF")
        (vdir / "voice.json").write_text(json.dumps({
            "name": "MollyVoice", "description": "flat, dry female voice", "ref_audio": "reference.wav"
        }), encoding="utf-8")
        self.assertEqual(drama_studio.voice_reference_path("MollyVoice", root=voices), str(ref_wav))
        self.assertIsNone(drama_studio.voice_reference_path("", root=voices))
        projects = self.root / "projects"
        with patch.object(drama, "PROJECTS_DIR", projects), \
                patch.object(drama_studio.voice_library, "VOICES_DIR", str(voices)):
            studio = drama_studio.DramaStudio()
            saved = studio.save_character("Pilot", {"name": "Molly", "gender": "female", "voice_name": "MollyVoice"})
            self.assertEqual(saved["voice"], "flat, dry female voice")

    def test_generate_plan_with_mocked_llm(self):
        projects = self.root / "projects"
        source = 'Molly stepped in. "Fine." she muttered.'
        plan_payload = {
            "characters": [{"id": "molly", "name": "Molly", "gender": "female",
                            "needs_reference": True, "appearance": "dark jacket",
                            "ref_tags": "black jacket, standing", "voice": "dry female voice"}],
            "segments": [{"segment": 1, "title": "Arrival", "duration": 4, "refs": ["molly"],
                          "palette": "soft slate", "tone": "dry parody",
                          "prompt": '<Image 1>, dark jacket, speaks in a dry female voice:\n"Fine."',
                          "dialogue": [{"speaker": "molly", "text": "Fine."}]}],
        }
        with patch.object(drama, "PROJECTS_DIR", projects), \
                patch.object(drama, "call_llm", return_value=json.dumps(plan_payload)) as mock_llm:
            studio = drama_studio.DramaStudio()
            plan = studio.generate_plan(source, "Pilot", {
                "provider": "OpenRouter", "llm_base_url": "https://openrouter.ai/api/v1",
                "llm_model": "openai/gpt-4o-mini", "comfy_url": "http://127.0.0.1:8188"}, "sk-test")
            self.assertEqual(len(plan["segments"]), 1)
            self.assertTrue((projects / "pilot" / "plan.json").is_file())
            self.assertIn("molly", studio.characters("Pilot"))
            self.assertTrue((projects / "pilot" / "references" / "characters.json").is_file())
            self.assertEqual(mock_llm.call_args.kwargs["api_key"], "sk-test")


if __name__ == "__main__":
    unittest.main()
