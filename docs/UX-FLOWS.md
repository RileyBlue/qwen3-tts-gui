# Qwen3-TTS Studio — UX flows and behavior

**Version:** 0.1 · **Status:** Proposal · **Initial focus:** Create Voice

## Primary behavioral outcome

**A person describes a desired voice → generates a sample → listens → saves or iterates.** A successful first-time experience should not require understanding hardware, attention backends, dtype, seeds, or chunk sizes.

## User scenarios

1. **First-time creator:** wants to hear a voice based on a short natural-language description.
2. **Iterating creator:** changes wording and compares the new result with prior output without accidental loss.
3. **Technical creator:** needs advanced settings for reproducibility or compatibility.
4. **Returning creator:** wants to reopen saved voices or create an audiobook; those features should remain reachable.

## Proposed Create Voice journey

| Phase | User intent | UI behavior | What must not happen |
| --- | --- | --- | --- |
| Describe | “What should it sound like?” | Focus on a labeled description field, optional examples, language if needed | Force hardware choices first |
| Generate | “Let me hear it.” | One visible primary `Generate voice` action | Competing orange navigation action |
| Prepare | “Is it working?” | Distinguish downloading and loading using actual status | Fake percent, duration, or countdown |
| Produce | “Is it still running?” | Show generation state and actual cancellation capability | Duplicate launches or unexplained freeze |
| Review | “Does this sound right?” | Play real output; accessible duration, save, download, regenerate | Pre-render fake waveform as if audio exists |
| Keep / refine | “I want to use this.” | Preserve current result, save explicitly, allow editable description | Destroy previous valid result unexpectedly |

## State machine (conceptual)

```text
IDLE
  └─ Generate → VALIDATING
       ├─ invalid → IDLE + inline field error
       └─ valid → PREPARING
                    ├─ first-time model absent → DOWNLOADING → LOADING
                    ├─ model cached → LOADING
                    ├─ failure → ERROR
                    └─ ready → GENERATING
                                  ├─ success → PREVIEW_READY
                                  ├─ failure → ERROR
                                  └─ safe cancel (if supported) → IDLE or PREVIEW_READY
PREVIEW_READY
  ├─ Play / Pause
  ├─ Save / Download
  ├─ Edit description (existing preview remains until replacement is ready)
  └─ Regenerate → VALIDATING
ERROR
  ├─ Retry → appropriate previous step
  └─ Reveal technical details → diagnostics
```

These states describe desired presentation. Whether Qt/model workers expose granular progress and cancellation must be verified before implementation.

## First-use model download

- Show `Downloading voice model` with a short explanation: `This is required the first time you use this model.`
- Display real progress if available (e.g. `402 MB of 3.83 GB`), not guessed time remaining.
- If only an indeterminate state is available, use an indeterminate indicator and explain what is happening.
- Once cached, do not announce a new download when only loading to memory.

## Errors and recovery

| Condition | User-facing approach | Recovery |
| --- | --- | --- |
| Unsupported `flash_attention_2` | `This acceleration option isn't available on this computer.` | Choose a verified supported setting or open advanced settings |
| Model download interrupted | `The model download didn't finish.` | Retry; reuse cached partial files if supported |
| Model load failure | `We couldn't load the voice model.` | Retry or inspect details |
| Insufficient memory | `This model needs more available memory.` | Explain realistic next steps; no unsupported backend claims |
| Audio output failure | `The sample was created, but playback failed.` | Preserve output and offer Download / Retry playback |
| Empty voice description | `Describe how you want the voice to sound.` | Focus field; preserve entered content |

Do not expose stack traces as primary user copy. Keep them available in expandable technical details or logs.

## Input and preset behavior

- Suggestions are optional, clearly secondary affordances.
- **Pending product decision:** selecting a suggestion should either fill an empty field or present a reversible replacement/append choice; avoid silent loss of user text.
- Preserve the field across generation, error and retry.
- If the current backend uses fixed audition text, clearly distinguish `Voice description` from `Sample text`. The preview text may live under a disclosure or as a compact secondary section, but must remain accessible if editable in the existing feature.
- Language must map to real model capabilities; do not expose languages unsupported by the loaded model.

## Follow-on flows (not in v0.1 implementation scope)

- **Clone Voice:** choose/reference source audio, understand recording constraints, enter audition text, generate, compare, save. Handle audio validation distinctly from model loading.
- **My Voices:** find saved voices, audition them, understand whether deletion is reversible, use a voice in a new generation.
- **Audiobooks:** enter or import text, select voice, review chunking/progress and outputs, resume/retry where supported.

## Research tasks

1. Generate a friendly narrator voice without assistance.
2. Change a description and listen to an alternative without losing previous work.
3. Find technical settings when explicitly asked to change attention mode.

Collect: task success, prompts needed, incorrect navigation, time to first interaction, time to first **audible** sample (reported separately from download/load), and perceived clarity of progress. Avoid inventing baseline metrics.
