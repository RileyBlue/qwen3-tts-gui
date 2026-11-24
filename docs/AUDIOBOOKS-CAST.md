# Audiobooks: Cast and voice compatibility

What the Audiobooks page checks before it renders, and why. The rules live in `cast_rules.py` (pure and read-only: it
reads `voice.json`, checks that files exist, reads a WAV header, and asks which packages this Python can import; it
loads no model and writes nothing). A segment still stores `"<engine>:<voice name>"` (Qwen uses `voice:`), so projects,
the segment cache and rendering are unchanged.

## The four states

| State | Meaning | Can it be assigned? | Does Render run? |
|---|---|---|---|
| Ready | Every requirement that can be checked locally is met. Model download and gated access are **not** checked. | yes | yes |
| Needs setup | Something outside the segment is missing and can be provided: a confirmed transcript, an F5 reference of 12 s or less, an optional package, F5's Python environment. | yes (so a project can be prepared) | no |
| Unsupported | The engine cannot speak this language, direction or `[emotion]` tags. | never newly; an existing one stays visible and editable | no |
| Missing | The voice, its reference file or its Qwen clone prompt is gone, or the key is not one this version knows. | kept as is | no |

Licence confirmations (XTTS, F5-TTS) and model downloads are not states: they happen at the first render, after the
pre-flight passes.

## What each engine needs (from `tts_generator.py` and `audiobook.render_segment`)

| | Qwen | XTTS | CSM | Dia | F5-TTS |
|---|---|---|---|---|---|
| Reads | clone prompt | reference WAV | reference WAV + `csm_ref_text` | reference WAV + `csm_ref_text` | `f5_ref_audio` + `f5_ref_text`, else reference WAV + `csm_ref_text` |
| Languages | all | 10 (not auto) | English | English | English, Chinese |
| Direction | yes | no | no | no | no |
| `[emotion]` tags | yes, if the voice has a description | no | no | no | no |
| Other | | optional `TTS` package | Transformers with CSM, gated model | Transformers with Dia | own Python (`F5_TTS_PYTHON`); reference <= 12 s |

A Qwen direction is sent as a leading `[tag]`, so it needs the same voice description a tag in the text does.

## How the page uses it

- Voice first: each saved voice is listed once; the Engine is a second control that shows every engine with its state.
- Bulk edits keep each segment's other half: a new voice keeps each engine, a new engine keeps each voice.
- Render and Preview run the pre-flight first, group problems by cause and offer to select the affected segments.
  Nothing is loaded and no licence is asked until it passes.
- Edits that make an assignment invalid (a direction, a tag, the Language) are flagged on the row and blocked at render,
  not prevented while typing.

## Not verified

- Qwen checks that happen after its model loads (a list-form clone prompt) are not covered by the parity test.
- Gated CSM access, the F5 environment named by `F5_TTS_PYTHON`, and model downloads are never checked.
- No real engine was run. Windows and Linux are unverified.
- F5's output sample rate comes from its worker; `compile_audio` raises on mixed rates, so a chapter that mixes F5 with
  another engine could fail at compile time. Not measured.
