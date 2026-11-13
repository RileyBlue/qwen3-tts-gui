"""UI-independent TTS generation logic for Qwen3-TTS.

Two ways to get one consistent voice:
  * VoiceClone   - saved/designed voice; emotion comes from per-emotion reference
                   clips generated with VoiceDesign and anchored to the voice's
                   speaker embedding (the Base model has no instruct input).
  * VoiceDesign  - used to create voices (and emotion references), not for long-form.
"""

import inspect
import json
import os

import numpy as np

from audio_utils import load_voice_clone_prompt, save_voice_clone_prompt, load_wav
from config import XTTS_LANGUAGES
from text_utils import chunk_text
import voice_library
from voice_lock import (
    EmotionPromptBank, parse_emotion_segments, seed_everything,
)


def _first(wavs):
    return np.asarray(wavs[0] if hasattr(wavs, "__getitem__") else wavs)


class StopRequested(Exception):
    pass


class TTSGenerator:
    def __init__(self, models, notify=print, should_stop=lambda: False):
        self.models = models
        self.notify = notify
        self.should_stop = should_stop

    # ---- helpers
    @staticmethod
    def chunk(text, min_words, max_words):
        if min_words and max_words and max_words > 0:
            return chunk_text(text, min_words=min_words, max_words=max_words)
        return [text]

    def _jobs(self, text, min_words, max_words):
        """Split on [emotion] tags first, then chunk each segment: [(emotion, chunk)]."""
        jobs = []
        for emotion, seg in parse_emotion_segments(text):
            for c in self.chunk(seg, min_words, max_words):
                jobs.append((emotion, c))
        return jobs

    # ---- VoiceDesign: voice creation
    def design_sample(self, description, language, seed):
        model = self.models.get("voice-design", self.notify)
        seed_everything(seed)
        wavs, sr = model.generate_voice_design(
            text=voice_library.SAMPLE_TEXT, language=language, instruct=description
        )
        return _first(wavs), sr

    def build_voice_prompt(self, wav, sr):
        base = self.models.get("voice-clone", self.notify)
        return base.create_voice_clone_prompt(
            ref_audio=(wav, sr), ref_text=voice_library.SAMPLE_TEXT, x_vector_only_mode=False
        )

    def build_clone_prompt(self, ref_audio, ref_text, x_vector_only=False):
        """Build a voice-clone prompt from reference audio without generating audio."""
        model = self.models.get("voice-clone", self.notify)
        return model.create_voice_clone_prompt(
            ref_audio=ref_audio, ref_text=ref_text or None, x_vector_only_mode=x_vector_only
        )

    def xtts(self, text, language, seed, temperature, voice_name):
        language_code = XTTS_LANGUAGES.get(language.lower())
        if not language_code:
            raise ValueError("XTTS requires an explicitly supported language; choose a language other than auto.")
        if any(emotion for emotion, _ in parse_emotion_segments(text)):
            raise ValueError("XTTS does not support emotion tags. Remove the tags before rendering.")
        path = voice_library.voice_dir(voice_name)
        with open(os.path.join(path, "voice.json"), encoding="utf-8") as voice_file:
            voice_config = json.load(voice_file)
        reference = os.path.join(path, voice_config.get("ref_audio", "reference.wav"))
        if not os.path.isfile(reference):
            raise ValueError(f"Reference audio not found for voice '{voice_name}': {reference}")
        if self.should_stop():
            return [], None
        model = self.models.get("xtts", self.notify)
        seed_everything(seed)
        split_sentences = model.synthesizer.split_into_sentences
        if "language" in inspect.signature(split_sentences).parameters:
            sentences = split_sentences(text, language=language_code)
        else:
            sentences = split_sentences(text)
        wavs = []
        for index, sentence in enumerate(sentences):
            if self.should_stop():
                return [], None
            self.notify(f"XTTS sentence {index + 1}/{len(sentences)}")
            wav = model.tts(text=sentence, speaker_wav=reference, language=language_code,
                            split_sentences=False, enable_text_splitting=True, temperature=temperature)
            wavs.append(np.asarray(wav, dtype=np.float32))
        if self.should_stop():
            return [], None
        return wavs, model.synthesizer.output_sample_rate

    def csm(self, text, language, seed, temperature, voice_name):
        if language.lower() != "english":
            raise ValueError("CSM supports English only; select English for this segment.")
        if any(emotion for emotion, _ in parse_emotion_segments(text)):
            raise ValueError("CSM does not support emotion tags. Remove the tags before rendering.")
        path = voice_library.voice_dir(voice_name)
        with open(os.path.join(path, "voice.json"), encoding="utf-8") as voice_file:
            voice_config = json.load(voice_file)
        transcript = (voice_config.get("csm_ref_text") or "").strip()
        if not transcript:
            raise ValueError("Set the exact reference transcript for this voice in the Voices tab before using CSM.")
        reference = os.path.join(path, voice_config.get("ref_audio", "reference.wav"))
        if not os.path.isfile(reference):
            raise ValueError(f"Reference audio not found for voice '{voice_name}': {reference}")
        if self.should_stop():
            return [], None
        import torch
        audio, sample_rate = load_wav(reference)
        if sample_rate != 24000:
            import torchaudio.functional
            audio = torchaudio.functional.resample(torch.from_numpy(audio), sample_rate, 24000).numpy()
        processor, model = self.models.get("csm", self.notify)
        seed_everything(seed)
        chunks = []
        for sentence in chunk_text(text, min_words=1, max_words=18):
            words = sentence.split()
            chunks.extend(" ".join(words[index:index + 18]) for index in range(0, len(words), 18))
        wavs = []
        for index, chunk in enumerate(chunks):
            if self.should_stop():
                return [], None
            self.notify(f"CSM chunk {index + 1}/{len(chunks)}")
            conversation = [
                {"role": "0", "content": [{"type": "text", "text": transcript},
                                          {"type": "audio", "path": audio}]},
                {"role": "0", "content": [{"type": "text", "text": chunk}]},
            ]
            inputs = processor.apply_chat_template(
                conversation, tokenize=True, return_dict=True, sampling_rate=24000
            ).to(model.device, dtype=model.dtype)
            budget = min(600, model.config.max_position_embeddings - inputs["input_ids"].shape[-1] - 1)
            if budget < 250:
                raise ValueError("CSM reference is too long for its context window; use a shorter recording and matching transcript.")
            with torch.inference_mode():
                result = model.generate(
                    **inputs, output_audio=True, return_dict_in_generate=True, max_new_tokens=budget,
                    do_sample=True, temperature=temperature, depth_decoder_do_sample=True,
                    depth_decoder_temperature=temperature,
                )
            final_frame = result.sequences[0, -1].detach().cpu().numpy()
            if not np.all(final_frame == model.config.codebook_eos_token_id):
                raise ValueError("CSM reached its audio limit before finishing; split the segment into shorter text.")
            wavs.append(result.audio[0].detach().cpu().float().numpy())
        if self.should_stop():
            return [], None
        return wavs, 24000

    def dia(self, text, language, seed, temperature, voice_name):
        if language.lower() != "english":
            raise ValueError("Dia supports English only; select English for this segment.")
        if any(emotion for emotion, _ in parse_emotion_segments(text)):
            raise ValueError("Dia does not support emotion tags. Remove the tags before rendering.")
        path = voice_library.voice_dir(voice_name)
        with open(os.path.join(path, "voice.json"), encoding="utf-8") as voice_file:
            voice_config = json.load(voice_file)
        transcript = (voice_config.get("csm_ref_text") or "").strip()
        if not transcript:
            raise ValueError("Set the exact reference transcript for this voice in the Voices tab before using Dia.")
        reference = os.path.join(path, voice_config.get("ref_audio", "reference.wav"))
        if not os.path.isfile(reference):
            raise ValueError(f"Reference audio not found for voice '{voice_name}': {reference}")
        if self.should_stop():
            return [], None
        import torch
        import torchaudio.functional
        audio, sample_rate = load_wav(reference)
        if sample_rate != 44100:
            audio = torchaudio.functional.resample(torch.from_numpy(audio), sample_rate, 44100).numpy()
        processor, model = self.models.get("dia", self.notify)
        seed_everything(seed)
        chunks = []
        for sentence in chunk_text(text, min_words=1, max_words=12):
            words = sentence.split()
            chunks.extend(" ".join(words[index:index + 12]) for index in range(0, len(words), 12))
        wavs = []
        for index, chunk in enumerate(chunks):
            if self.should_stop():
                return [], None
            self.notify(f"Dia chunk {index + 1}/{len(chunks)}")
            with torch.inference_mode():
                inputs = processor(text=f"[S1] {transcript} [S1] {chunk}", audio=audio,
                                   sampling_rate=44100, return_tensors="pt").to(model.device)
                prompt_len = processor.get_audio_prompt_len(inputs["decoder_attention_mask"])
                budget = min(1400, model.config.decoder_config.max_position_embeddings -
                             inputs["decoder_attention_mask"].shape[-1] - max(model.config.delay_pattern) - 1)
                if budget < 500 or inputs["input_ids"].shape[-1] > model.config.encoder_config.max_position_embeddings:
                    raise ValueError("Dia reference or transcript is too long; use a shorter recording and matching transcript.")
                output = model.generate(**inputs, max_new_tokens=budget, do_sample=True, temperature=temperature)
                if output.shape[1] - inputs["decoder_input_ids"].shape[1] >= budget:
                    raise ValueError("Dia reached its audio limit before finishing; split the segment into shorter text.")
                wave = processor.batch_decode(output, audio_prompt_len=prompt_len)[0].float().cpu()
                wavs.append(torchaudio.functional.resample(wave, 44100, 24000).numpy())
        if self.should_stop():
            return [], None
        return wavs, 24000

    def f5(self, text, language, seed, temperature, voice_name):
        if language.lower() not in ("english", "chinese"):
            raise ValueError("F5-TTS supports English and Chinese; select a supported language.")
        if any(emotion for emotion, _ in parse_emotion_segments(text)):
            raise ValueError("F5-TTS does not support emotion tags. Remove the tags before rendering.")
        path = voice_library.voice_dir(voice_name)
        with open(os.path.join(path, "voice.json"), encoding="utf-8") as voice_file:
            voice_config = json.load(voice_file)
        specific_reference = voice_config.get("f5_ref_audio")
        transcript = (voice_config.get("f5_ref_text") if specific_reference else
                      voice_config.get("csm_ref_text")) or ""
        transcript = transcript.strip()
        if not transcript:
            raise ValueError("Set the exact reference transcript for this voice in the Voices tab before using F5-TTS.")
        reference = os.path.join(path, specific_reference or voice_config.get("ref_audio", "reference.wav"))
        if not os.path.isfile(reference):
            raise ValueError(f"Reference audio not found for voice '{voice_name}': {reference}")
        audio, sample_rate = load_wav(reference)
        if not len(audio) or len(audio) / sample_rate > 12:
            raise ValueError("F5-TTS needs a reference recording no longer than 12 seconds; set one in the Voices tab.")
        if self.should_stop():
            return [], None
        model = self.models.get("f5", self.notify)
        chunks = []
        for sentence in chunk_text(text, min_words=1, max_words=40):
            words = sentence.split()
            chunks.extend(" ".join(words[index:index + 40]) for index in range(0, len(words), 40))
        wavs, output_rate = [], None
        for index, chunk in enumerate(chunks):
            if self.should_stop():
                return [], None
            self.notify(f"F5-TTS chunk {index + 1}/{len(chunks)}")
            wav, sample_rate = model.infer(reference, transcript, chunk, seed)
            if output_rate is not None and sample_rate != output_rate:
                raise ValueError("F5-TTS returned inconsistent sample rates.")
            output_rate = sample_rate
            wavs.append(np.asarray(wav, dtype=np.float32))
        if self.should_stop():
            return [], None
        return wavs, output_rate

    # ---- VoiceClone
    def clone(self, text, language, min_words, max_words, seed=None, temperature=0.7,
              voice_name=None, ref_audio="", ref_text=None, x_vector_only=False,
              reuse_prompt=None, save_prompt=None, description=""):
        voice_cfg = voice_dir = None
        if voice_name:
            voice_cfg, anchor_prompt, voice_dir = voice_library.load_voice(voice_name)
        elif not ref_audio and not reuse_prompt:
            raise ValueError("Pick a saved voice, or provide reference audio / a reusable prompt.")

        model = self.models.get("voice-clone", self.notify)
        if x_vector_only:
            ref_text = None
        if not voice_cfg:
            if reuse_prompt:
                anchor_prompt = load_voice_clone_prompt(reuse_prompt)
            else:
                anchor_prompt = model.create_voice_clone_prompt(
                    ref_audio=ref_audio, ref_text=ref_text or None, x_vector_only_mode=x_vector_only
                )

        if save_prompt:
            save_voice_clone_prompt(save_prompt, anchor_prompt)
            self.notify(f"Voice-clone prompt saved to: {save_prompt}")

        jobs = self._jobs(text, min_words, max_words)
        bank = None
        if any(e for e, _ in jobs):
            if not isinstance(anchor_prompt, list):
                raise ValueError("Emotion tags need a prompt saved as a list of items; rebuild the prompt.")
            description = voice_cfg["description"] if voice_cfg else (description or "").strip()
            if not description:
                raise ValueError("Describe the voice (description box) to use [emotion] tags.")
            bank = EmotionPromptBank(
                vd_model=self.models.get("voice-design", self.notify),
                base_model=model,
                anchor_item=anchor_prompt[0],
                voice_description=description,
                language=language,
                # A saved voice reuses its own seed so emotion references reproduce exactly.
                seed=voice_cfg["seed"] if voice_cfg else (seed or 0),
                cache_dir=os.path.join(voice_dir, "emotions") if voice_dir else None,
            )

        # Lower sampling temperature keeps timbre steadier across long text.
        gen_kwargs = dict(temperature=temperature, subtalker_temperature=temperature)
        out, sr = [], None
        for i, (emotion, t) in enumerate(jobs):
            if self.should_stop():
                break
            self.notify(f"Chunk {i+1}/{len(jobs)}" + (f" [{emotion}]" if emotion else ""))
            prompt = bank.get(emotion) if bank else anchor_prompt
            seed_everything(seed)
            wavs, sr = model.generate_voice_clone(
                text=t, language=language, voice_clone_prompt=prompt, **gen_kwargs
            )
            out.append(_first(wavs))
        return out, sr
