# Qwen3-TTS GUI — Design documentation

**Status:** Proposed / not yet implemented  
**Version:** 0.1  
**Scope:** Initial Create Voice redesign, with reusable guidance for future screens  
**Platform:** Desktop, PySide6 / Qt  
**Related project:** https://github.com/ephrisian/qwen3-tts-gui  
**Contribution fork:** https://github.com/RileyBlue/qwen3-tts-gui

This directory records the proposed user-experience direction and implementation guardrails. **These documents are not proof that the redesigned UI or automatic GPU selection exists in the code.** The existing application remains the functional source of truth until changes are implemented and verified.

## Start here

| Document | Purpose |
| --- | --- |
| [DESIGN.md](DESIGN.md) | Product principles, visual foundations, interaction contract, and implementation rules |
| [UX-FLOWS.md](UX-FLOWS.md) | Behavior goals, user journeys, interface states, and suggested copy |
| [COMPONENTS.md](COMPONENTS.md) | Proposed component inventory and PySide6 implementation patterns |
| [DESIGN-QA.md](DESIGN-QA.md) | Acceptance checks, accessibility checks, and contribution verification |

## Design intent

A person should be able to describe a voice, generate a sample, listen, and save it without understanding PyTorch, dtype, FlashAttention, or model internals. Expert controls remain accessible when needed.

**Core rule:** Navigation communicates location. Actions communicate intent. They must not compete for attention.

## Adoption plan

1. Review the concepts with the maintainer before broad changes.
2. Implement **Create Voice** in a focused branch, preserving behavior and model logic.
3. Verify the baseline end-to-end before expanding to Clone Voice, My Voices, or Audiobooks.
4. Capture before/after screenshots and document actual behavior in the PR.
5. Update these documents when implementation decisions change.

## Source-of-truth boundaries

- Existing `qt_app.py` and `qt_audiobook.py` contain the current Qt UI.
- Existing model and audio modules own synthesis, loading, storage, and playback behavior.
- `.hf_cache/`, `output/`, and `voices/` contain local/generated assets; never include them in design commits.
- The current desktop UI and the proposed visuals may differ. Follow implemented code for operational behavior; follow approved design documentation for new changes.

## Open decisions

- Confirm preferred accent color and typography with the maintainer (tokens below are proposals).
- Decide whether suggestion chips prefill or append to a voice description; do not silently overwrite user input.
- Decide whether preview audio can be regenerated without replacing a previously saved voice.
- Confirm operating-system support and minimum window size with QA.
- Test whether automatic backend selection is feasible separately; it is not a prerequisite for initial UI layout changes.
