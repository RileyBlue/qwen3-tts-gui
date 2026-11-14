# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

Recorded as `web` because Impeccable has no desktop value. This is a **desktop PySide6 / Qt Widgets application**, not a website: styling is Qt Style Sheets (`theme.py`), there is no browser, DOM or CSS. Treat web-specific guidance (landing pages, responsive breakpoints, browser live mode) as not applicable. Confirmed with the user.

## Users

Narration and audiobook creators producing spoken audio (narration, audiobooks, character voices) on their own computer. They typically design or clone a voice once, then reuse it for long-form renders. Confirmed with the user.

Scenarios described in `docs/UX-FLOWS.md` (proposal, not research): a first-time creator who wants to hear a voice from a short description; an iterating creator comparing alternatives without losing earlier work; a technical creator who needs advanced controls; a returning creator reopening saved voices or building an audiobook.

## Product Purpose

Qwen3-TTS Studio is a desktop front end for the Qwen3-TTS text-to-speech model. A person describes a voice in words, hears a sample, saves it, and then uses it for voice cloning with emotion direction and for rendering whole chapters as audiobooks. Success is a first audible sample, and later a consistent saved voice, without needing to understand PyTorch, dtype or attention backends (`docs/DESIGN.md`).

## Positioning

Voice creation, voice reuse and long-form rendering in one local desktop tool, built around a saved voice that keeps one stable identity: a designed sample is converted to a reusable clone prompt, and `[emotion]` tags are performed through per-emotion reference clips anchored to that voice (`README.md`, `FEATURES.md`, `tts_generator.py`).

## Operating Context

- Runs entirely on the user's machine. Model weights download once from Hugging Face into the project-local `.hf_cache`; generated audio goes to `output/`, saved voices to `voices/<name>/` (`voice.json`, `reference.wav`, `clone_prompt.npz`).
- Workflows: Create Voice (VoiceDesign), Clone Voice (reference audio or saved voice, `[tag]` direction), My Voices (library), Audiobooks (segments, per-segment voice and direction, compiled WAV), plus a CLI benchmark.
- Long-running model work runs on a background thread; only clone and audiobook renders can be stopped part-way.

## Capabilities and Constraints

Confirmed by the user as fixed for future work:

- **Fully local.** No account, no cloud service in the product.
- **macOS first; Windows and Linux must keep working.** First target is Apple Silicon (M3 Pro, 18 GB). The existing Windows/NVIDIA (`install.bat`) and Linux (`manage_qwen.sh`) paths must not regress.
- **No hardware claims until verified.** Do not present automatic device selection or MPS support until real inference is verified on that hardware (design contract decision D-006). Current defaults (`cuda`/`cpu`, `bfloat16`, `flash_attention_2`) are unchanged.
- **Voice-generation behavior is frozen.** UI work must not change synthesis, model loading, saved-voice format or output paths without separate review.

Facts from the code, not separately confirmed:

- Voice creation uses a fixed audition passage; seed does not guarantee the same voice for new text.
- Qwen3-TTS has no markup for stressing single words; emphasis comes from wording and punctuation.
- Supported languages: auto, English, Chinese, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian.

Undecided: whether model selection belongs in the main flow; whether suggestion chips fill or append (currently append, reversible); whether a previously saved voice can be regenerated without replacing it; minimum supported window size beyond the current 1024 × 640.

## Brand Commitments

Product name shown in the app: "Qwen3-TTS Studio". No logo, wordmark or brand voice has been established. The accent color, typography and tokens in `docs/DESIGN.md` are explicitly proposals awaiting maintainer review, not approved brand.

## Evidence on Hand

- `docs/` (DESIGN, COMPONENTS, UX-FLOWS, DESIGN-QA, README): proposed design contract, not proof of implemented behavior.
- `docs/references/create-voice-target.png`: a concept image. It shows items the product does not have or must not claim (an "M3 Pro · Auto" device pill, fixed `0:00 / 0:12` playback position, a 500-character limit, an account avatar). Do not treat those as requirements.
- Absent, and not to be invented: user research, usability metrics, testimonials, customers, benchmark timings across hardware, and any claim that Apple GPU (MPS) inference works.

## Product Principles

1. The path from description to an audible sample is the product; everything else is secondary until the first sample exists.
2. Technical controls stay available but not prominent; a creator is never forced to make a hardware decision to get a first result.
3. Never discard the user's text or a valid generated result, including after a failure.
4. Show only real state: no invented durations, waveforms, progress or hardware capability.
5. Preserve what already works on every supported platform; redesign the interface, not the synthesis.

## Accessibility & Inclusion

`docs/DESIGN.md` proposes WCAG 2.2 AA for text and interaction states: full keyboard operation with visible focus, persistent labels, accessible names for icon-only controls, status and errors exposed to assistive technology, no color-only state, and legibility at 125–200% scaling. This is a design target; screen-reader and OS-scaling checks have not been run.
