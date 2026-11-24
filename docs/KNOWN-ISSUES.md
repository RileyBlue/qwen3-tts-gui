# Known issues and verification limits

Facts about the voice library and Clone Voice that are true today and not yet fixed. Verified means exercised by
`python launch.py test` or `python tests_ui.py` on macOS.

## Voice library: failure-safe replace and the folder lock

- Replacing a voice builds the new one in `voices/.pending/`, swaps it in with renames and deletes the old copy
  last. A failure at any step leaves the old voice as it was (verified by injected failures at each step).
- A lock file, `voices/.voices.lock`, stops two app windows from saving at once (verified with a real second
  process on macOS). The operating system drops it when the process ends. A save waits up to 10 seconds.
- **Not verified:** locking on Windows (`msvcrt` path), and Linux.
- **Not verified:** surviving a real crash or power loss in the middle of a swap. Crashes were simulated by
  making renames fail, never by killing the process. Files are flushed before the swap, but durability across power
  loss is best effort and depends on the platform and file system.
- Renaming a directory onto an existing one is not atomic across platforms, so the swap is two renames with a
  rollback and a recovery step (`recover_interrupted_saves`), not a single operation.

## Seed: "Random" and the emotion samples

- In Clone Voice, **Random** means the speech itself is not reseeded.
- `[tag]` emotion samples are different: for a recording or prompt file, `clone()` seeds them with `seed or 0`,
  so with Random they always use seed 0. For a saved voice they use the seed stored in its `voice.json`.
- Save as voice therefore stores the Fixed value, or 0 for Random, which is what `clone()` would have used.
- The label "Random" does not describe the tag samples. Changing that would change generation, so it is tracked
  here as a separate issue.

## Saved voice metadata

- Voices saved before the transcript change keep `ref_text` and `sample_text` set to the standard passage even
  when they were cloned from a recording. Only the transcript editor reads `ref_text` (as a prefill).
- The `usage` note in `voice.json` says to rebuild a prompt from `reference.wav` and `ref_text`, which is wrong for
  voices saved with an empty transcript (X-vector only).

## Preview (Clone Voice)

- The finished file is written and read back on the interface thread. Measured on macOS: a 60 minute mono 24 kHz
  file takes about 0.4 s to write (existing behaviour) and about 0.2 s to read back.
- Screen reader behaviour (VoiceOver and others) is not verified.

## Reference audio formats and dependencies

Checked by hand on macOS (librosa 1.0.1, libsndfile 1.2.2), not by the test suites unless stated.

- The model reads reference audio through `librosa`, which here decodes with `soundfile` (libsndfile) only. WAV,
  FLAC, MP3 and OGG loaded through both `librosa.load` and `load_wav`. M4A did not: this librosa version has no
  `audioread` fallback.
- The file pickers offer only formats the installed libsndfile reports, so **M4A is never offered**, even where an
  older librosa with `audioread` could decode it. Choosing an unreadable file explains the problem before any model
  loads (the header check is covered by the test suites).
- MP3 and OGG are offered only when the installed libsndfile reports them. Behaviour on Windows and Linux, and with
  other libsndfile versions, is not verified.
- `load_wav` (shared) reads 8/16/32-bit PCM WAV with the standard library as before and falls back to `soundfile`
  for float or 24-bit WAV, FLAC and MP3 (covered by the test suites). That also applies to Import voice from audio
  and the F5 reference. Clone's Save as voice still requires a `.wav` file.
- `soundfile` reaches the app only through `qwen_tts`. From reading the code (not exercised), the app falls back to
  offering WAV only without it. The tests import it directly.
- From reading the code (not exercised): a float WAV with values above full scale is clipped in the saved
  `reference.wav`, while the prompt is built from the unclipped array.

## Real model

- Clone Voice has not been run with the real Base model on this machine. All evidence so far comes from a stand-in
  model.
