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
