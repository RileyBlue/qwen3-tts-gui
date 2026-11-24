# Qwen3-TTS GUI

A desktop GUI for the Qwen3-TTS text-to-speech model, supporting VoiceDesign (voice creation), VoiceClone (voice cloning), and audiobook rendering.

## Requirements

- **Conda** (Miniconda or Anaconda) — required to create the isolated Python environment.
- **NVIDIA GPU** with CUDA support recommended (torch is installed from the cu121 wheel).
- **Windows**: use `install.bat` (see below).
- **Linux/macOS**: use `./manage_qwen.sh install` (see below).

## Installation

### Windows (Conda required)

1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/) if you do not already have Conda.
2. Open a terminal (PowerShell or Command Prompt) in this project folder.
3. Run the Windows installer:

   ```bat
   install.bat
   ```

   This creates a Conda environment named `qwen3-tts` (Python 3.10) and installs all dependencies:
   - torch / torchvision / torchaudio (cu121)
   - qwen-tts
   - flash-attn
   - pygame
   - PySide6-Essentials

### Linux / macOS

1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/) if you do not already have Conda.
2. Open a terminal in this project folder.
3. Run:

   ```bash
   On Linux, double-click `Start Qwen3-TTS.desktop` in the project folder to launch
   the GUI without entering a command. The launcher opens a terminal window so startup
   messages and errors remain visible. If your file manager asks, choose “Allow
   Launching” or “Trust and Launch”.
   ./manage_qwen.sh install
   ```

   This performs the same steps as the Windows installer using the shell script.

## Usage

### Windows

After installation, activate the environment and launch the GUI:

```bat
conda activate qwen3-tts
python launch.py
```

Or run directly from any shell:

```bat
conda run -n qwen3-tts python launch.py
```

### Linux / macOS

```bash
./manage_qwen.sh run
```

Or:

```bash
conda activate qwen3-tts
python launch.py
```

## HTTP API

The GUI automatically starts a local HTTP API at `http://127.0.0.1:8765`. It uses
the same lazily loaded models and saved voice library as the GUI, so model weights
are not loaded twice. Use `python launch.py --no-api` to disable it, or change the
address with `--host` and `--port`.

To run without the GUI (and without importing PySide6):

```bash
python launch.py --api-only
# Custom model settings are optional:
python launch.py --api-only --host 127.0.0.1 --port 8765 --device cuda --dtype bfloat16 --attn flash_attention_2
```

Available endpoints:

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/docs` | Interactive Swagger UI |
| `GET` | `/openapi.json` | OpenAPI 3.0 specification |
| `POST` | `/mcp` | Model Context Protocol endpoint for compatible harnesses |
| `GET` | `/health` | Server status and whether generation is busy |
| `GET` | `/v1/voices` | Saved voices and their metadata |
| `GET` | `/v1/config` | Languages and generation defaults |
| `POST` | `/v1/tts/design` | Generate a sample from a voice description |
| `POST` | `/v1/tts/clone` | Generate speech with a saved voice or reference audio |

Generation endpoints accept JSON and return a WAV file directly. They also save
the WAV under `output/`; `X-Output-Path`, `X-Sample-Rate`, and `X-Seed` response
headers provide its metadata.

```bash
# Design a voice (text is optional; the standard audition text is the default).
curl -o designed.wav http://127.0.0.1:8765/v1/tts/design \
  -H 'Content-Type: application/json' \
  -d '{"description":"A warm, calm female narrator","text":"Hello from the API.","language":"english","seed":42}'

# Clone a saved voice. Use GET /v1/voices to find available names.
curl -o cloned.wav http://127.0.0.1:8765/v1/tts/clone \
  -H 'Content-Type: application/json' \
  -d '{"voice":"Narrator","text":"A longer passage to speak.","language":"english","seed":42,"temperature":0.7}'
```

Clone requests also accept `min_words`, `max_words`, `ref_audio`, `ref_text`,
`x_vector_only`, `reuse_prompt`, `save_prompt`, and `description`. Supply either
`voice`, `ref_audio`, or `reuse_prompt`. The API processes one generation at a
time and returns HTTP 409 while the GUI or another API request is busy.

Python scripts can use the standard library without an API client dependency:

```python
import json, urllib.request
payload = json.dumps({"voice": "Narrator", "text": "Hello from Python."}).encode()
request = urllib.request.Request("http://127.0.0.1:8765/v1/tts/clone", data=payload,
                                 headers={"Content-Type": "application/json"})
with urllib.request.urlopen(request) as response, open("speech.wav", "wb") as output:
    output.write(response.read())
```

## Swagger and MCP

Open `http://127.0.0.1:8765/docs` for interactive Swagger documentation, or use
`http://127.0.0.1:8765/openapi.json` with another OpenAPI client. Swagger UI loads
its browser assets from jsDelivr, so displaying the page requires internet access;
the OpenAPI JSON and API itself remain local.

Harnesses supporting streamable HTTP MCP can connect directly to
`http://127.0.0.1:8765/mcp`. A typical MCP configuration is:

```json
{
  "mcpServers": {
    "qwen3-tts": {
      "url": "http://127.0.0.1:8765/mcp"
    }
  }
}
```

The endpoint implements `initialize`, `ping`, `tools/list`, and `tools/call` using
MCP protocol version `2025-03-26`. It exposes `list_voices`, `get_config`,
`voice_design`, and `voice_clone`. Generation tools return base64-encoded
`audio/wav` content plus text metadata containing the output path, sample rate,
and seed. The server is stateless and does not create MCP sessions or open a
server-initiated event stream.

The API has no authentication and is intended for trusted local scripts and
harnesses. Do not bind it to a non-loopback address unless you protect it with
appropriate network controls.

## Drama Studio (ComfyUI + LLM)

The **Drama Studio** tab integrates `minimax_drama.py` with local or remote
ComfyUI image/video workflows (`workflows/image-Gen_api.json` and
`workflows/video_minimax_h3_r2v_api.json`).

1. **Settings (LLM & ComfyUI)**: Open **Settings** and configure the LLM
   provider (`OpenRouter`, `OpenAI`, or `Custom / local`), base URL, model,
   session API key (or `OPENROUTER_API_KEY` / `OPENAI_API_KEY`), and ComfyUI URL
   (`http://127.0.0.1:8188` or a remote server). Saved settings exclude the API
   key.
2. **Project**: Select an existing project or create a new one at the top of
   Drama Studio before managing its cast or script; project assets live under
   `projects/<project>/`.
3. **Cast**: Create or edit project characters in
   `projects/<project>/references/characters.json`, assign and preview a saved
   Qwen voice, and attach a reference image (or tags for ComfyUI reference
   generation).
4. **Script & Render**: Paste prose or a story prompt with verbatim dialogue,
   generate and review the segment plan JSON, generate any missing reference
   images into `projects/<project>/references/`, and render/concatenate the
   final vertical video under `projects/<project>/`. Final stitching requires
   `ffmpeg` and `ffprobe` on `PATH`.

## Project commands

| Command | Description |
|---------|-------------|
| `install` | Create the Conda environment and install dependencies |
| `run` | Launch the GUI (requires the environment to be installed) |
| `remove` | Remove the Conda environment |
| `cleanup` | Clean pip cache, temp files, and reset install checkpoints |
| `purge` | Remove the Conda environment and all project-generated files |

Windows: `install.bat` handles installation; use `conda activate qwen3-tts && python launch.py` to run.

Linux/macOS: `./manage_qwen.sh {install|run|remove|cleanup|purge}`

## Optional XTTS-v2 audiobook voices

XTTS-v2 is an optional audiobook backend; Qwen remains available for voice design,
cloning, and audiobook rendering. Install the maintained Coqui package in the
existing `qwen3-tts` environment (review the proposed dependency changes first):

```bash
conda run -n qwen3-tts python -m pip install --dry-run "coqui-tts>=0.27.4,<0.29"
conda run -n qwen3-tts python -m pip install "coqui-tts>=0.27.4,<0.29"
conda run -n qwen3-tts python -m pip check
```

In the Audiobook tab, select `XTTS: <voice>` for a narrator, dialogue, or individual
segment. XTTS uses the voice's saved reference WAV; it does not use Qwen's clone
prompt. Choose an explicit supported language instead of `auto`. Direction and
emotion tags are not supported for XTTS segments. XTTS uses its own sentence
splitting, and stop requests take effect between sentences.

The first render downloads the XTTS model and asks you to accept the
[Coqui Public Model License](https://tts-hub.github.io/cpml) or confirm you have
a commercial license. XTTS runs in its default precision; the global dtype,
attention, and word-chunk controls do not apply to XTTS. Mixed-engine chapters
require matching sample rates.

## Optional CSM-1B audiobook voices

CSM uses the Transformers package already installed by Qwen. Before first use,
accept the access conditions at [sesame/csm-1b](https://huggingface.co/sesame/csm-1b)
and log in with a Hugging Face account that has access. On Linux/macOS, from the
project directory:

```bash
conda activate qwen3-tts
export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"
hf auth login
```

On Windows Command Prompt, use `set HF_HOME=%CD%\.hf_cache` instead of `export`.
Enter the token directly into the login prompt; the token is stored under
`HF_HOME` unless `HF_TOKEN_PATH` is set.

In the Voices tab, select a voice and choose **Reference transcript**. Confirm that the
text exactly matches the entire reference recording, including for previously
imported voices. Then select `CSM: <voice>` in the Audiobook tab and choose
English. CSM uses the saved reference WAV and confirmed transcript, not the Qwen
clone prompt. Direction and emotion tags are unsupported. CSM splits text into
short chunks and stops between chunks; a long reference may need to be replaced
with a shorter recording and matching transcript to fit the context window.

CSM uses the global device and CUDA dtype settings, SDPA attention, and its own
chunking. CPU inference uses float32. Voice similarity and long-form quality may
vary, and CSM is not the same fine-tuned model as Sesame's online demo.

## Optional F5-TTS and Dia audiobook voices

Both engines condition each chunk on the saved voice's reference recording and
matching transcript. This encourages a consistent voice, but does not guarantee
identical timbre across chunks or engines. Select `F5-TTS: <voice>` or
`Dia: <voice>` in the Audiobook tab. Neither engine supports the app's direction
or emotion tags. F5 supports English and Chinese; Dia supports English.

In **Voices**, use **Reference transcript** to confirm the exact words in the
full reference WAV. Dia uses this reference. For F5, the reference must be at
most 12 seconds: if the saved recording is longer, use **F5 reference** to
select a short PCM WAV and enter its exact transcript. F5 uses that separate
reference when present. Its pretrained weights are licensed under
[CC BY-NC 4.0](https://huggingface.co/SWivid/F5-TTS), not for commercial use
without permission.

Dia uses the Transformers package installed with Qwen; it downloads the ungated
Dia checkpoint and audio codec on first use. Dia outputs 44.1 kHz, converted to
24 kHz for audiobook mixing. The model authors report around 10 GB VRAM for the
full model. The device and CUDA dtype settings apply; Dia uses SDPA attention.

Install F5 in a separate environment so its dependencies do not replace Qwen's.
Choose a suitable PyTorch/torchaudio build for your hardware following the
[F5-TTS installation guide](https://github.com/SWivid/F5-TTS#installation),
then install F5 and FFmpeg. For example, on Linux/macOS:

```bash
conda create -n f5-tts python=3.11
conda activate f5-tts
conda install ffmpeg
python -m pip install "f5-tts==1.1.22"
export F5_TTS_PYTHON="$(python -c 'import sys; print(sys.executable)')"
conda activate qwen3-tts
python launch.py
```

On Windows, set `F5_TTS_PYTHON` to the F5 environment's Python executable
before launching the GUI. The helper loads F5 once per render group and exits
when switching engines. F5 uses its own sampling settings; the GUI temperature
control does not apply. Both engines stop between chunks.

## Inference benchmark

List saved voices, then compare one voice, text, and seed across every backend
combination for the selected device, dtype, and attention options. The default
keeps chunk size fixed at 250 words so backend results are directly comparable:

```bash
python launch.py benchmark --list-voices
python launch.py benchmark --voice narrator --text-file test-text.md --seed 42 \
   --dtypes bfloat16,float16,float32 \
   --attentions flash_attention_2,sdpa,eager
python launch.py benchmark --voice narrator --text-file test-text.md --seed 42 \
   --matrix chunking --chunk-sizes 0,100,250,400
```

Backend mode defaults to the first 50 words, no warm-up passes, and one measured
pass per configuration to keep the nine-way sweep quick. Increase `--warmups` and
`--repeats` for more stable timing, or use `--max-input-words 0` to benchmark the
entire input. Chunking mode uses the full file by default. Use `--matrix chunking`
to compare chunk sizes using just the baseline backend; `--matrix full` tests all
backend and chunk-size combinations. Use `--devices`, `--dtypes`, and
`--attentions` to select the options to sweep; `--matrix backends` (the default)
tries every combination of those lists at the same chunk size. Unsupported
combinations are recorded as failures. The benchmark randomizes configuration
order, records model-load time separately, and reports median RTF, run-to-run
spread, and change versus baseline. Results and successful WAVs go to a timestamped
folder under `output/`. Since the default backend sweep measures each configuration
once, treat it as a screening run; use repeats for close comparisons.

## Notes

- Qwen models are downloaded on first use into the project-local `.hf_cache` directory (via `HF_HOME`). Coqui manages its own model cache.
- Generated audio is written to `output/` and saved voices to `voices/`.
- At most two models stay in memory at once.
- The GUI is built with PySide6; all model logic is UI-independent (`tts_generator.py`, `audiobook.py`, `voice_library.py`, `voice_lock.py`, `model_loader.py`).
