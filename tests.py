"""Unit tests for Qwen3-TTS GUI modules."""

import os
import sys
import tempfile
import wave

import numpy as np

from config import OUTPUT_DIR
from text_utils import chunk_text, split_text
from audio_utils import write_wav, normalize_wavs, save_outputs, save_voice_clone_prompt, load_voice_clone_prompt


def _project_temp_path(suffix=".wav"):
    """Return a temp file path inside the project folder (output dir)."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fd, path = tempfile.mkstemp(suffix=suffix, dir=OUTPUT_DIR)
    os.close(fd)
    return path


# ------------------------------------------------------------------ audio utils

def _test_write_wav():
    """Test that write_wav produces a valid WAV file."""
    sr = 16000
    duration = 0.5  # seconds
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave

    path = _project_temp_path()

    try:
        write_wav(path, audio, sr)

        # Verify by reading back
        with wave.open(path, "rb") as wf:
            assert wf.getnchannels() == 1, "Expected mono channel"
            assert wf.getsampwidth() == 2, "Expected 16-bit samples"
            assert wf.getframerate() == sr, f"Expected sample rate {sr}"
            n_frames = wf.getnframes()
            expected_frames = int(sr * duration)
            assert abs(n_frames - expected_frames) < 10, (
                f"Expected ~{expected_frames} frames, got {n_frames}"
            )
        print("PASS: write_wav produces valid WAV file")
    finally:
        os.unlink(path)


def _test_write_wav_various_inputs():
    """Test write_wav handles various input types (scalar, 0-d, 1-d, multi-channel)."""
    sr = 16000

    test_cases = [
        ("scalar float", np.float32(0.5)),
        ("0-d array", np.array(0.5)),
        ("1-d array", 0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 0.5, int(sr * 0.5)))),
        ("multi-channel", np.stack([
            0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 0.5, int(sr * 0.5))),
            0.3 * np.sin(2 * np.pi * 880 * np.linspace(0, 0.5, int(sr * 0.5))),
        ], axis=1)),
    ]

    for name, audio in test_cases:
        path = _project_temp_path()
        try:
            write_wav(path, audio, sr)
            with wave.open(path, "rb") as wf:
                assert wf.getnchannels() == 1, f"{name}: Expected mono channel"
                assert wf.getsampwidth() == 2, f"{name}: Expected 16-bit samples"
                assert wf.getframerate() == sr, f"{name}: Expected sample rate {sr}"
                n_frames = wf.getnframes()
                assert n_frames > 0, f"{name}: Expected non-empty audio"
            print(f"PASS: write_wav handles {name}")
        finally:
            os.unlink(path)


def _test_normalize_wavs():
    """Test normalize_wavs normalizes various wavs input types correctly."""

    # Test 1: numpy scalar
    wavs = normalize_wavs(np.float32(0.5))
    assert len(wavs) == 1, f"Expected 1, got {len(wavs)}"
    assert wavs[0].shape == (1,), f"Expected shape (1,), got {wavs[0].shape}"
    print("PASS: normalize_wavs normalizes numpy scalar")

    # Test 2: 0-d array
    wavs = normalize_wavs(np.array(0.5))
    assert len(wavs) == 1
    assert wavs[0].shape == (1,)
    print("PASS: normalize_wavs normalizes 0-d array")

    # Test 3: 1-d array
    wavs = normalize_wavs(np.array([0.1, 0.2, 0.3]))
    assert len(wavs) == 1
    assert wavs[0].shape == (3,)
    print("PASS: normalize_wavs normalizes 1-d array")

    # Test 4: list of arrays
    wavs = normalize_wavs([np.array([0.1, 0.2]), np.array([0.3, 0.4])])
    assert len(wavs) == 2
    print("PASS: normalize_wavs normalizes list of arrays")

    # Test 5: numpy generic scalar
    wavs = normalize_wavs(np.float64(0.75))
    assert len(wavs) == 1
    assert wavs[0].shape == (1,)
    print("PASS: normalize_wavs normalizes numpy generic scalar")


# ------------------------------------------------------------------ text utils

def _test_split_text():
    """Test split_text returns the text as a single-item list."""
    texts = split_text("Hello, World, Test")
    assert texts == ["Hello, World, Test"]
    print("PASS: split_text returns single-item list")


def _test_split_text_empty():
    """Test split_text returns empty list for empty input."""
    assert split_text("") == []
    assert split_text(None) == []
    print("PASS: split_text handles empty input")


def _test_chunk_text():
    """Test chunk_text splits long text at sentence boundaries."""
    text = "This is sentence one. This is sentence two! Is this sentence three? Yes it is."
    chunks = chunk_text(text, min_words=5, max_words=10)
    assert len(chunks) >= 1
    # Each chunk should be a non-empty string
    for chunk in chunks:
        assert len(chunk.strip()) > 0
    print("PASS: chunk_text splits text correctly")


def _test_chunk_text_empty():
    """Test chunk_text returns empty list for empty input."""
    assert chunk_text("") == []
    assert chunk_text(None) == []
    print("PASS: chunk_text handles empty input")


def _test_chunk_text_single_sentence():
    """Test chunk_text keeps a single short sentence as one chunk."""
    chunks = chunk_text("Hello world.", min_words=5, max_words=10)
    assert len(chunks) == 1
    assert chunks[0] == "Hello world."
    print("PASS: chunk_text keeps single short sentence intact")


# ------------------------------------------------------------------ NPZ / voice clone prompt loading

def _test_npz_load_allow_pickle():
    """Test that NPZ files containing object arrays load with allow_pickle=True."""
    path = _project_temp_path(suffix=".npz")
    try:
        # Save an object array (e.g. a dict with a numpy array inside)
        obj = {"ref_spk_embedding": np.array([0.1, 0.2, 0.3]), "other": "value"}
        np.savez(path, **obj)

        # Loading with allow_pickle=True should succeed
        data = np.load(path, allow_pickle=True)
        assert set(data.files) == {"ref_spk_embedding", "other"}
        print("PASS: NPZ object arrays load with allow_pickle=True")
    finally:
        os.unlink(path)


def _test_npz_numeric_keys_convert_to_list():
    """Test that NPZ saved from a list (numeric string keys) converts back to a list."""
    path = _project_temp_path(suffix=".npz")
    try:
        # Simulate saving a list (as gui_app.py does): convert to {str(i): v} then np.savez(**dict)
        prompt_list = [np.array([0.1, 0.2]), np.array([0.3, 0.4])]
        prompt_dict = {str(i): v for i, v in enumerate(prompt_list)}
        np.savez(path, **prompt_dict)

        data = np.load(path, allow_pickle=True)
        voice_clone_prompt = {k: data[k] for k in data.files}

        # Numeric keys should be detected and converted back to a list
        assert voice_clone_prompt and all(k.isdigit() for k in voice_clone_prompt.keys())
        converted = [voice_clone_prompt[k] for k in sorted(voice_clone_prompt.keys(), key=int)]
        assert len(converted) == 2
        assert np.array_equal(converted[0], np.array([0.1, 0.2]))
        assert np.array_equal(converted[1], np.array([0.3, 0.4]))
        print("PASS: NPZ numeric-string keys convert back to list")
    finally:
        os.unlink(path)


def _test_npz_named_keys_stay_dict():
    """Test that NPZ saved with named keys stays a dict (no conversion)."""
    path = _project_temp_path(suffix=".npz")
    try:
        prompt_dict = {"ref_spk_embedding": np.array([0.1, 0.2]), "other": np.array([0.3])}
        np.savez(path, **prompt_dict)

        data = np.load(path, allow_pickle=True)
        voice_clone_prompt = {k: data[k] for k in data.files}

        # Named keys are not all digits, so no conversion
        assert not all(k.isdigit() for k in voice_clone_prompt.keys())
        assert "ref_spk_embedding" in voice_clone_prompt
        assert np.array_equal(voice_clone_prompt["ref_spk_embedding"], np.array([0.1, 0.2]))
        print("PASS: NPZ named keys stay as dict")
    finally:
        os.unlink(path)


def _test_npz_empty_files():
    """Test that an empty NPZ does not crash the conversion logic."""
    path = _project_temp_path(suffix=".npz")
    try:
        np.savez(path)  # empty
        data = np.load(path, allow_pickle=True)
        voice_clone_prompt = {k: data[k] for k in data.files}
        assert voice_clone_prompt == {}
        print("PASS: empty NPZ handled gracefully")
    finally:
        os.unlink(path)


# ------------------------------------------------------------------ voice clone prompt helpers

class _FakePrompt:
    """Stand-in for objects returned by create_voice_clone_prompt (has ref_code attr)."""
    def __init__(self, embedding):
        self.ref_code = embedding


def _test_save_load_voice_clone_prompt_preserves_ref_code():
    """Test that save/load round-trips objects with ref_code attribute intact."""
    path = _project_temp_path(suffix=".npz")
    try:
        prompt = [_FakePrompt(np.array([0.1, 0.2, 0.3])), _FakePrompt(np.array([0.4, 0.5]))]
        save_voice_clone_prompt(path, prompt)

        loaded = load_voice_clone_prompt(path)
        assert isinstance(loaded, list)
        assert len(loaded) == 2
        assert hasattr(loaded[0], "ref_code"), "Loaded object lost ref_code attribute"
        assert np.array_equal(loaded[0].ref_code, np.array([0.1, 0.2, 0.3]))
        assert np.array_equal(loaded[1].ref_code, np.array([0.4, 0.5]))
        print("PASS: save/load voice clone prompt preserves ref_code")
    finally:
        os.unlink(path)


def _test_save_load_voice_clone_prompt_dict():
    """Test that save/load round-trips a dict of objects with ref_code intact."""
    path = _project_temp_path(suffix=".npz")
    try:
        prompt = {"ref_spk_embedding": _FakePrompt(np.array([0.1, 0.2]))}
        save_voice_clone_prompt(path, prompt)

        loaded = load_voice_clone_prompt(path)
        assert isinstance(loaded, dict)
        assert "ref_spk_embedding" in loaded
        assert hasattr(loaded["ref_spk_embedding"], "ref_code")
        assert np.array_equal(loaded["ref_spk_embedding"].ref_code, np.array([0.1, 0.2]))
        print("PASS: save/load voice clone prompt dict preserves ref_code")
    finally:
        os.unlink(path)


# ------------------------------------------------------------------ save_outputs

def _test_save_outputs_single():
    """Test save_outputs writes a single WAV file."""
    sr = 16000
    audio = 0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 0.5, int(sr * 0.5)))
    path = _project_temp_path()
    try:
        result = save_outputs(audio, sr, path)
        assert result == path
        assert os.path.exists(path)
        with wave.open(path, "rb") as wf:
            assert wf.getnframes() > 0
        print("PASS: save_outputs writes single WAV")
    finally:
        if os.path.exists(path):
            os.unlink(path)


def _test_save_outputs_stop_requested():
    """Test save_outputs skips writing when stop_requested is True."""
    path = _project_temp_path()
    os.unlink(path)  # remove pre-created file so we can verify nothing is written
    try:
        result = save_outputs(np.array([0.1, 0.2]), 16000, path, stop_requested=True)
        assert result == path
        assert not os.path.exists(path)
        print("PASS: save_outputs skips when stop_requested")
    finally:
        if os.path.exists(path):
            os.unlink(path)


def _test_audiobook_split():
    import audiobook
    segs = audiobook.split_narration_dialogue(
        'The rain fell. "Where?" she asked. \u201cHere,\u201d he said. Done.', "N", "D")
    assert [s["kind"] for s in segs] == ["narration", "dialogue", "narration", "dialogue", "narration"], segs
    assert segs[1]["text"] == '"Where?"' and segs[1]["voice"] == "D"
    assert audiobook.split_narration_dialogue("", "N", "D") == []
    a, b = audiobook.split_segment_at(segs[0], 8)
    assert a["text"] == "The rain" and b["text"] == "fell."
    assert audiobook.merge_segments(a, b)["text"] == "The rain fell."
    assert audiobook.apply_direction({"text": "Hi", "direction": "sad"}) == "[sad] Hi"
    print("PASS: audiobook split/merge/direction")


def _test_audiobook_compile():
    import audiobook
    sr = 1000
    quiet = (np.ones(500, dtype=np.float32) * 0.01, sr)
    loud = (np.ones(500, dtype=np.float32) * 0.5, sr)
    out, osr = audiobook.compile_audio([quiet, loud], pause_ms=100, match_loudness=True)
    assert osr == sr and len(out) == 500 + 100 + 500
    assert abs(out[0] - out[-1]) < 1e-4, "loudness should be matched"
    assert np.all(out[500:600] == 0), "pause should be silent"
    out2, _ = audiobook.compile_audio([quiet, loud], pause_ms=0, match_loudness=False)
    assert len(out2) == 1000
    print("PASS: audiobook compile")


def _test_audiobook_cache_and_grouping():
    import audiobook
    calls = []

    class Gen:
        def clone(self, **kw):
            calls.append("clone")
            return [np.zeros(10, dtype=np.float32)], 24000

        def xtts(self, **kw):
            calls.append("xtts")
            return [np.ones(10, dtype=np.float32)], 24000

        def csm(self, **kw):
            calls.append("csm")
            return [np.full(10, 2, dtype=np.float32)], 24000

        def f5(self, **kw):
            calls.append("f5")
            return [np.full(10, 3, dtype=np.float32)], 24000

        def dia(self, **kw):
            calls.append("dia")
            return [np.full(10, 4, dtype=np.float32)], 44100

    cache = tempfile.mkdtemp(dir=os.path.dirname(os.path.abspath(__file__)))
    try:
        settings = dict(language="english", min_words=0, max_words=0, seed=1, temperature=0.7)
        seg = audiobook.new_segment("Hi.", "voice:Ryan")
        audiobook.render_segment(Gen(), seg, settings, cache_dir=cache)
        audiobook.render_segment(Gen(), seg, settings, cache_dir=cache)
        assert calls == ["clone"], "second render should hit the cache"
        audiobook.render_segment(Gen(), dict(seg, text="Changed."), settings, cache_dir=cache)
        assert calls == ["clone", "clone"]
        xtts_seg = dict(seg, voice="xtts:Ryan")
        wav, _ = audiobook.render_segment(Gen(), xtts_seg, settings, cache_dir=cache)
        assert calls == ["clone", "clone", "xtts"] and np.all(wav == 1)
        audiobook.render_segment(Gen(), xtts_seg, settings, cache_dir=cache)
        assert calls == ["clone", "clone", "xtts"]
        csm_seg = dict(seg, voice="csm:Ryan")
        wav, _ = audiobook.render_segment(Gen(), csm_seg, settings, cache_dir=cache)
        assert calls == ["clone", "clone", "xtts", "csm"] and np.all(wav == 2)
        audiobook.render_segment(Gen(), csm_seg, settings, cache_dir=cache)
        assert calls == ["clone", "clone", "xtts", "csm"]
        for engine, value in (("f5", 3), ("dia", 4)):
            engine_seg = dict(seg, voice=f"{engine}:Ryan")
            wav, _ = audiobook.render_segment(Gen(), engine_seg, settings, cache_dir=cache)
            assert calls[-1] == engine and np.all(wav == value)
            count = len(calls)
            audiobook.render_segment(Gen(), engine_seg, settings, cache_dir=cache)
            assert len(calls) == count
            try:
                audiobook.render_segment(Gen(), dict(engine_seg, direction="sad"), settings, cache_dir=cache)
                assert False, "unsupported directions should be rejected"
            except ValueError as exc:
                assert "direction" in str(exc)
        try:
            audiobook.render_segment(Gen(), dict(csm_seg, direction="sad"), settings, cache_dir=cache)
            assert False, "CSM directions should be rejected"
        except ValueError as exc:
            assert "direction" in str(exc)
        try:
            audiobook.render_segment(Gen(), dict(xtts_seg, direction="sad"), settings, cache_dir=cache)
            assert False, "XTTS directions should be rejected"
        except ValueError as exc:
            assert "direction" in str(exc)
        from unittest.mock import patch
        order = []

        def fake_render(gen, segment, settings):
            order.append(segment["voice"])
            return np.zeros(2), 24000

        segments = [dict(seg, voice=voice) for voice in
                    ("csm:Ryan", "xtts:Ryan", "csm:Ryan", "voice:Ryan", "xtts:Ryan")]
        with patch.object(audiobook, "render_segment", fake_render):
            results = audiobook.render_chapter(Gen(), segments, settings, notify=lambda message: None)
        assert len(results) == 5
        assert order == ["csm:Ryan", "csm:Ryan", "voice:Ryan", "xtts:Ryan", "xtts:Ryan"]
    finally:
        import shutil
        shutil.rmtree(cache, ignore_errors=True)
    print("PASS: audiobook segment cache")


def _test_xtts_generation():
    import json
    from unittest.mock import patch
    import voice_library
    import tts_generator

    class Synthesizer:
        output_sample_rate = 24000

        def split_into_sentences(self, text, language):
            assert language == "en"
            return ["First.", "Second."]

    class Model:
        synthesizer = Synthesizer()

        def tts(self, **kwargs):
            calls.append(kwargs)
            return [0.1, 0.2]

    class Models:
        def get(self, kind, notify):
            assert kind == "xtts"
            return Model()

    calls = []
    with tempfile.TemporaryDirectory() as root:
        path = os.path.join(root, "Ryan")
        os.makedirs(path)
        reference = os.path.join(path, "reference.wav")
        write_wav(reference, np.zeros(20), 24000)
        with open(os.path.join(path, "voice.json"), "w", encoding="utf-8") as voice_file:
            json.dump({"name": "Ryan", "ref_audio": "reference.wav"}, voice_file)
        with patch.object(voice_library, "VOICES_DIR", root), patch.object(tts_generator, "seed_everything"):
            import audiobook
            assert audiobook.voice_choices() == [("Qwen: Ryan", "voice:Ryan"), ("XTTS: Ryan", "xtts:Ryan"),
                                                 ("CSM: Ryan", "csm:Ryan"), ("F5-TTS: Ryan", "f5:Ryan"),
                                                 ("Dia: Ryan", "dia:Ryan")]
            segment = audiobook.new_segment("Hello.", "xtts:Ryan")
            settings = {"language": "english"}
            key_before = audiobook.segment_cache_key(segment, settings)
            write_wav(reference, np.ones(20) * 0.1, 24000)
            assert audiobook.segment_cache_key(segment, settings) != key_before
            project_path = os.path.join(root, "project.json")
            audiobook.save_project(project_path, [segment, dict(segment, voice="voice:Ryan")], {})
            restored, _ = audiobook.load_project(project_path)
            assert [item["voice"] for item in restored] == ["xtts:Ryan", "voice:Ryan"]
            generator = tts_generator.TTSGenerator(Models(), notify=lambda message: None)
            wavs, sr = generator.xtts("First. Second.", "english", 42, 0.7, "Ryan")
            assert len(wavs) == 2 and sr == 24000
            assert all(call["speaker_wav"] == reference and call["language"] == "en" for call in calls)
            assert all(call["temperature"] == 0.7 and call["enable_text_splitting"] for call in calls)
            with patch.object(Synthesizer, "split_into_sentences", lambda self, text: [text]):
                calls.clear()
                wavs, sr = generator.xtts("Legacy.", "english", 42, 0.7, "Ryan")
                assert len(wavs) == 1 and calls[0]["text"] == "Legacy." and sr == 24000
            calls.clear()
            generator.should_stop = lambda: len(calls) == 1
            assert generator.xtts("First. Second.", "english", 42, 0.7, "Ryan") == ([], None)
            assert len(calls) == 1
            for text, language in (("Hello", "auto"), ("[sad] Hello", "english")):
                try:
                    generator.xtts(text, language, 42, 0.7, "Ryan")
                    assert False, "unsupported XTTS input should fail"
                except ValueError:
                    pass
    print("PASS: XTTS reference, language, and cancellation")


def _test_csm_generation():
    import contextlib
    import json
    import types
    from unittest.mock import patch
    import audiobook
    import voice_library
    import tts_generator

    class Tensor:
        def __init__(self, value):
            self.value = np.asarray(value)
            self.shape = self.value.shape

        def __getitem__(self, index):
            return Tensor(self.value[index])

        def detach(self):
            return self

        def cpu(self):
            return self

        def float(self):
            return self

        def numpy(self):
            return self.value

    class Inputs(dict):
        def to(self, device, dtype):
            assert device == "cpu" and dtype == "float32"
            return self

    class Processor:
        input_length = 100

        def apply_chat_template(self, conversation, **kwargs):
            conversations.append(conversation)
            assert kwargs == {"tokenize": True, "return_dict": True, "sampling_rate": 24000}
            return Inputs(input_ids=Tensor(np.zeros((1, self.input_length))))

    class Model:
        device = "cpu"
        dtype = "float32"
        config = types.SimpleNamespace(max_position_embeddings=2048, codebook_eos_token_id=0)
        complete = True

        def generate(self, **kwargs):
            calls.append(kwargs)
            end = 0 if self.complete else 1
            return types.SimpleNamespace(sequences=Tensor([[[1, 1], [end, end]]]),
                                         audio=[Tensor([0.1, 0.2])])

    class Models:
        def get(self, kind, notify):
            assert kind == "csm"
            return processor, model

    calls = []
    conversations = []
    processor = Processor()
    model = Model()
    with tempfile.TemporaryDirectory() as root:
        path = os.path.join(root, "Ryan")
        os.makedirs(path)
        reference = os.path.join(path, "reference.wav")
        write_wav(reference, np.zeros(2400), 24000)
        with open(os.path.join(path, "voice.json"), "w", encoding="utf-8") as voice_file:
            json.dump({"name": "Ryan", "ref_text": "Unconfirmed text", "ref_audio": "reference.wav"}, voice_file)
        fake_torch = types.SimpleNamespace(inference_mode=contextlib.nullcontext, from_numpy=Tensor)
        with (patch.object(voice_library, "VOICES_DIR", root),
              patch.object(tts_generator, "seed_everything"), patch.dict(sys.modules, {"torch": fake_torch})):
            generator = tts_generator.TTSGenerator(Models(), notify=lambda message: None)
            try:
                generator.csm("Hello.", "english", 42, 0.7, "Ryan")
                assert False, "an unconfirmed transcript should fail"
            except ValueError as exc:
                assert "transcript" in str(exc)
            segment = audiobook.new_segment("Hello.", "csm:Ryan")
            key_before = audiobook.segment_cache_key(segment, {})
            voice_library.set_csm_reference_text("Ryan", "  This matches the recording.  ")
            assert audiobook.segment_cache_key(segment, {}) != key_before
            key_before = audiobook.segment_cache_key(segment, {})
            write_wav(reference, np.ones(2400) * 0.1, 24000)
            assert audiobook.segment_cache_key(segment, {}) != key_before
            wavs, sr = generator.csm("Hello there. Goodbye.", "english", 42, 0.7, "Ryan")
            assert sr == 24000 and len(wavs) == 1
            assert conversations[0][0]["content"][0]["text"] == "This matches the recording."
            assert len(conversations[0][0]["content"][1]["path"]) == 2400
            assert calls[0]["max_new_tokens"] == 600 and calls[0]["temperature"] == 0.7
            functional = types.ModuleType("torchaudio.functional")
            audio_module = types.ModuleType("torchaudio")
            audio_module.functional = functional

            def resample(audio, source_rate, target_rate):
                assert source_rate == 48000 and target_rate == 24000
                return Tensor(audio.value[::2])

            functional.resample = resample
            write_wav(reference, np.zeros(4800), 48000)
            with patch.dict(sys.modules, {"torchaudio": audio_module, "torchaudio.functional": functional}):
                generator.csm("Hello.", "english", 42, 0.7, "Ryan")
            assert len(conversations[-1][0]["content"][1]["path"]) == 2400
            write_wav(reference, np.zeros(2400), 24000)
            for text, language in (("Hello", "auto"), ("[sad] Hello", "english")):
                try:
                    generator.csm(text, language, 42, 0.7, "Ryan")
                    assert False, "unsupported CSM input should fail"
                except ValueError:
                    pass
            processor.input_length = 1800
            try:
                generator.csm("Hello.", "english", 42, 0.7, "Ryan")
                assert False, "oversized CSM context should fail"
            except ValueError as exc:
                assert "context" in str(exc)
            processor.input_length = 100
            model.complete = False
            try:
                generator.csm("Hello.", "english", 42, 0.7, "Ryan")
                assert False, "truncated CSM audio should fail"
            except ValueError as exc:
                assert "audio limit" in str(exc)
            model.complete = True
            calls.clear()
            generator.should_stop = lambda: len(calls) == 1
            assert generator.csm(" ".join(["word"] * 25), "english", 42, 0.7, "Ryan") == ([], None)
            assert len(calls) == 1
    print("PASS: CSM reference, generation, context, and cancellation")


def _test_f5_generation_and_worker():
    import contextlib
    import io
    import json
    import types
    from unittest.mock import patch
    import audiobook
    import f5_worker
    import voice_library
    import tts_generator

    calls = []

    class Model:
        def infer(self, reference, transcript, text, seed):
            calls.append((reference, transcript, text, seed))
            return np.ones(10), 24000

    class Models:
        def get(self, kind, notify):
            assert kind == "f5"
            return Model()

    with tempfile.TemporaryDirectory() as root:
        path = os.path.join(root, "Ryan")
        os.makedirs(path)
        reference = os.path.join(root, "short.wav")
        write_wav(reference, np.zeros(2400), 24000)
        with open(os.path.join(path, "voice.json"), "w", encoding="utf-8") as voice_file:
            json.dump({"name": "Ryan", "ref_text": "Unconfirmed"}, voice_file)
        with patch.object(voice_library, "VOICES_DIR", root):
            generator = tts_generator.TTSGenerator(Models(), notify=lambda message: None)
            try:
                generator.f5("Hello.", "english", 42, 0.7, "Ryan")
                assert False, "unconfirmed reference must fail"
            except ValueError as exc:
                assert "transcript" in str(exc)
            segment = audiobook.new_segment("Hello.", "f5:Ryan")
            key_before = audiobook.segment_cache_key(segment, {})
            voice_library.set_f5_reference("Ryan", reference, "Reference words.")
            assert audiobook.segment_cache_key(segment, {}) != key_before
            wavs, sr = generator.f5("Hello.", "english", 42, 0.7, "Ryan")
            assert len(wavs) == 1 and sr == 24000
            assert calls[0] == (os.path.join(path, "f5_reference.wav"), "Reference words.", "Hello.", 42)
            key_before = audiobook.segment_cache_key(segment, {})
            voice_library.set_f5_reference("Ryan", reference, "Changed words.")
            assert audiobook.segment_cache_key(segment, {}) != key_before
            calls.clear()
            generator.should_stop = lambda: len(calls) == 1
            assert generator.f5(" ".join(["word"] * 50), "english", 42, 0.7, "Ryan") == ([], None)
            assert len(calls) == 1
            long_reference = os.path.join(root, "long.wav")
            write_wav(long_reference, np.zeros(12 * 24000 + 1), 24000)
            try:
                voice_library.set_f5_reference("Ryan", long_reference, "Words")
                assert False, "long reference must fail"
            except ValueError as exc:
                assert "12 seconds" in str(exc)

        class FakeF5:
            def __init__(self, device):
                assert device == "cpu"

            def infer(self, **kwargs):
                assert kwargs["ref_text"] == "Reference words." and kwargs["seed"] == 42
                return np.array([0.1, 0.2]), 24000, None

        package = types.ModuleType("f5_tts")
        api = types.ModuleType("f5_tts.api")
        api.F5TTS = FakeF5
        output_path = os.path.join(root, "result.npz")
        request = dict(reference=reference, transcript="Reference words.", text="Hello.",
                       seed=42, output_path=output_path)
        output = io.StringIO()
        with (patch.dict(sys.modules, {"f5_tts": package, "f5_tts.api": api}),
              patch.object(sys, "stdin", io.StringIO(json.dumps(request) + "\n")),
              contextlib.redirect_stdout(output)):
            f5_worker.run_worker("cpu")
        responses = [json.loads(line[len(f5_worker._PREFIX):]) for line in output.getvalue().splitlines()]
        assert responses == [{"ok": True}, {"ok": True, "sample_rate": 24000}]
        with np.load(output_path) as data:
            assert np.allclose(data["wav"], [0.1, 0.2])
        package_dir = os.path.join(root, "f5_tts")
        os.makedirs(package_dir)
        with open(os.path.join(package_dir, "__init__.py"), "w", encoding="utf-8") as package_file:
            package_file.write("")
        with open(os.path.join(package_dir, "api.py"), "w", encoding="utf-8") as api_file:
            api_file.write(
                "import numpy as np\n"
                "class F5TTS:\n"
                "    def __init__(self, device):\n"
                "        print('loading fake F5')\n"
                "    def infer(self, **kwargs):\n"
                "        return np.array([0.3, 0.4]), 24000, None\n"
            )
        with patch.dict(os.environ, {"F5_TTS_PYTHON": sys.executable, "PYTHONPATH": root}):
            worker = f5_worker.F5Worker("cpu")
            try:
                wav, rate = worker.infer(reference, "Reference words.", "Hello.", 42)
                assert rate == 24000 and np.allclose(wav, [0.3, 0.4])
            finally:
                worker.close()
    print("PASS: F5 reference, caching, cancellation, and worker protocol")


def _test_dia_generation():
    import contextlib
    import json
    import types
    from unittest.mock import patch
    import voice_library
    import tts_generator

    class Tensor:
        def __init__(self, value):
            self.value = np.asarray(value)
            self.shape = self.value.shape

        def float(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.value

    class Inputs(dict):
        def to(self, device):
            assert device == "cpu"
            return self

    class Processor:
        def __call__(self, **kwargs):
            prompts.append(kwargs)
            return Inputs(input_ids=Tensor(np.zeros((1, 50))),
                          decoder_input_ids=Tensor(np.zeros((1, 100, 9))),
                          decoder_attention_mask=Tensor(np.zeros((1, 115))))

        def get_audio_prompt_len(self, mask):
            return 100

        def batch_decode(self, output, audio_prompt_len):
            assert audio_prompt_len == 100
            return [Tensor(np.ones(441))]

    class Model:
        device = "cpu"
        config = types.SimpleNamespace(
            decoder_config=types.SimpleNamespace(max_position_embeddings=3072),
            encoder_config=types.SimpleNamespace(max_position_embeddings=1024),
            delay_pattern=[0, 15])

        def generate(self, **kwargs):
            calls.append(kwargs)
            return Tensor(np.zeros((1, 200, 9)))

    class Models:
        def get(self, kind, notify):
            assert kind == "dia"
            return Processor(), Model()

    calls, prompts, rates = [], [], []
    functional = types.ModuleType("torchaudio.functional")
    audio_module = types.ModuleType("torchaudio")
    audio_module.functional = functional

    def resample(audio, source_rate, target_rate):
        rates.append((source_rate, target_rate))
        return Tensor(np.zeros(round(len(audio.value) * target_rate / source_rate)))

    functional.resample = resample
    fake_torch = types.SimpleNamespace(inference_mode=contextlib.nullcontext, from_numpy=Tensor)
    with tempfile.TemporaryDirectory() as root:
        path = os.path.join(root, "Ryan")
        os.makedirs(path)
        write_wav(os.path.join(path, "reference.wav"), np.zeros(2400), 24000)
        with open(os.path.join(path, "voice.json"), "w", encoding="utf-8") as voice_file:
            json.dump({"name": "Ryan", "csm_ref_text": "Reference words."}, voice_file)
        with (patch.object(voice_library, "VOICES_DIR", root), patch.object(tts_generator, "seed_everything"),
              patch.dict(sys.modules, {"torch": fake_torch, "torchaudio": audio_module,
                                       "torchaudio.functional": functional})):
            generator = tts_generator.TTSGenerator(Models(), notify=lambda message: None)
            wavs, sr = generator.dia("First sentence.", "english", 42, 0.7, "Ryan")
            assert sr == 24000 and len(wavs) == 1 and len(wavs[0]) == 240
            assert prompts[0]["text"] == "[S1] Reference words. [S1] First sentence."
            assert rates == [(24000, 44100), (44100, 24000)]
            assert calls[0]["max_new_tokens"] == 1400 and calls[0]["temperature"] == 0.7
            calls.clear()
            generator.should_stop = lambda: len(calls) == 1
            assert generator.dia(" ".join(["word"] * 20), "english", 42, 0.7, "Ryan") == ([], None)
            assert len(calls) == 1
    print("PASS: Dia reference, resampling, and cancellation")


def _test_engine_model_lifecycle():
    import importlib.util
    import types
    from unittest.mock import patch
    import f5_worker

    fake_torch = types.ModuleType("torch")
    fake_torch.float32, fake_torch.float16, fake_torch.bfloat16 = "float32", "float16", "bfloat16"
    fake_torch.cuda = types.SimpleNamespace(empty_cache=lambda: None)
    qwen = types.ModuleType("qwen_tts")
    qwen.Qwen3TTSModel = object
    events = []

    class Worker:
        def __init__(self, device):
            events.append(("start", device))

        def close(self):
            events.append(("close",))

    class AudioTokenizer:
        def to(self, device):
            events.append(("codec", device))
            return self

        def eval(self):
            return self

    class Processor:
        audio_tokenizer = AudioTokenizer()

        @classmethod
        def from_pretrained(cls, model_id):
            return cls()

    class DiaModel:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            assert kwargs["torch_dtype"] == "float32"
            return cls()

        def to(self, device):
            return self

        def eval(self):
            return self

    transformers = types.ModuleType("transformers")
    transformers.AutoProcessor = Processor
    transformers.DiaForConditionalGeneration = DiaModel
    module_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_loader.py")
    spec = importlib.util.spec_from_file_location("_test_engine_model_loader", module_path)
    module = importlib.util.module_from_spec(spec)
    with (patch.dict(sys.modules, {"torch": fake_torch, "qwen_tts": qwen, "transformers": transformers}),
          patch.object(f5_worker, "F5Worker", Worker)):
        spec.loader.exec_module(module)
        manager = module.ModelManager(lambda: ("cpu", "bfloat16", "sdpa"))
        with patch.object(module, "load_model", return_value=object()):
            manager.get("voice-clone")
            first = manager.get("f5")
            assert manager.get("f5") is first
            manager.get("dia")
            assert events == [("start", "cpu"), ("close",), ("codec", "cpu")]
            assert list(manager._models) == ["dia"]
            manager.get("voice-design")
            assert list(manager._models) == ["voice-design"]
    print("PASS: engine model lifecycle")


def _test_voice_library_roundtrip():
    import torch
    import voice_library
    root = tempfile.mkdtemp(dir=os.path.dirname(os.path.abspath(__file__)))
    try:
        prompt = [_FakePrompt(torch.zeros(3))]
        wav = np.zeros(2400, dtype=np.float32)
        cfg = voice_library.save_voice("My Voice!", wav, 24000, prompt, "warm", "english", 7, root=root)
        assert voice_library.list_voices(root) == ["My Voice!"]
        loaded, p, path = voice_library.load_voice("My Voice!", root)
        assert loaded["seed"] == 7 and loaded["description"] == "warm" and cfg["sample_text"] == voice_library.SAMPLE_TEXT
        assert os.path.exists(os.path.join(path, "reference.wav"))
        try:
            voice_library.save_voice("My Voice!", wav, 24000, prompt, "x", "english", 1, root=root)
            raise AssertionError("expected FileExistsError")
        except FileExistsError:
            pass
        voice_library.delete_voice("My Voice!", root)
        assert voice_library.list_voices(root) == []
    finally:
        import shutil
        shutil.rmtree(root, ignore_errors=True)
    print("PASS: voice library round trip")


def _test_emotion_tags_and_seed():
    import torch
    from voice_lock import parse_emotion_segments, parse_seed, seed_everything
    assert parse_emotion_segments("a [angry] b [whisper] c [1] d") == [
        (None, "a"), ("angry", "b"), ("whisper", "c [1] d")]
    assert parse_seed("Random", "5") is None and parse_seed("Fixed", "5") == 5
    seed_everything(3); x = torch.rand(2)
    seed_everything(3); assert torch.equal(x, torch.rand(2))
    print("PASS: emotion tags and seeding")


def _test_benchmark_helpers():
    from benchmark import (
        automatic_chunk_sizes, build_configurations, chunk_bounds, prepare_input_text,
        real_time_factor,
    )
    assert chunk_bounds(0) == (0, 0)
    assert chunk_bounds(250) == (200, 250)
    assert chunk_bounds(1) == (1, 1)
    assert real_time_factor(12, 60) == 0.2
    assert real_time_factor(1, 0) is None
    assert automatic_chunk_sizes("one two three four five six") == [0, 3, 2]
    sample = "First sentence has four words. Second sentence has five words."
    assert prepare_input_text(sample, 5) == "First sentence has four words."
    assert prepare_input_text(sample, 0) == sample
    one_factor = build_configurations(["cuda"], ["bf16", "fp16"], ["flash", "sdpa"],
                                      [0, 3], order_seed=5)
    assert len(one_factor) == 4
    assert len(set(one_factor)) == 4
    backends = build_configurations(["cuda"], ["bf16", "fp16"], ["flash", "sdpa"],
                                    [250, 400], matrix="backends", order_seed=5)
    assert len(backends) == 4
    assert {config[3] for config in backends} == {250}
    assert {(config[1], config[2]) for config in backends} == {
        ("bf16", "flash"), ("bf16", "sdpa"), ("fp16", "flash"), ("fp16", "sdpa")
    }
    chunking = build_configurations(["cuda"], ["bf16"], ["sdpa"], [0, 250, 400],
                                   matrix="chunking", order_seed=5)
    assert {config[3] for config in chunking} == {0, 250, 400}
    full = build_configurations(["cuda"], ["bf16", "fp16"], ["flash", "sdpa"],
                                [0, 3], matrix="full", order_seed=5)
    assert len(full) == 8
    print("PASS: benchmark helpers")


# ------------------------------------------------------------------ runner

def run_tests():
    """Run all unit tests."""
    print("Running unit tests...")
    _test_write_wav()
    _test_write_wav_various_inputs()
    _test_normalize_wavs()
    _test_split_text()
    _test_split_text_empty()
    _test_chunk_text()
    _test_chunk_text_empty()
    _test_chunk_text_single_sentence()
    _test_npz_load_allow_pickle()
    _test_npz_numeric_keys_convert_to_list()
    _test_npz_named_keys_stay_dict()
    _test_npz_empty_files()
    _test_save_load_voice_clone_prompt_preserves_ref_code()
    _test_save_load_voice_clone_prompt_dict()
    _test_save_outputs_single()
    _test_save_outputs_stop_requested()
    _test_audiobook_split()
    _test_audiobook_compile()
    _test_audiobook_cache_and_grouping()
    _test_xtts_generation()
    _test_csm_generation()
    _test_dia_generation()
    _test_f5_generation_and_worker()
    _test_engine_model_lifecycle()
    _test_voice_library_roundtrip()
    _test_emotion_tags_and_seed()
    _test_benchmark_helpers()
    print("All tests passed.")


if __name__ == "__main__":
    run_tests()
