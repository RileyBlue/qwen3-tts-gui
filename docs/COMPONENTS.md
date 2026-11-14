# Qwen3-TTS Studio — Proposed Qt component contracts

**Status:** Design proposal. Names below are conceptual; they are **not claims about existing Python classes**.

## Component inventory

| Component | Qt Widgets building blocks | Key contract |
| --- | --- | --- |
| `AppShell` | `QMainWindow`, `QWidget`, `QHBoxLayout` | Sidebar, work area, resizing, clear content hierarchy |
| `SidebarNavItem` | `QPushButton` (checkable) or accessible equivalent | Selected location is neutral, not visually confused with primary CTA |
| `PageHeader` | `QLabel`, `QVBoxLayout` | One heading + at most one short description |
| `VoiceDescriptionField` | `QLabel`, `QTextEdit`, helper/counter labels | Persistent label, editable, visible focus, preserved input |
| `SuggestionChip` | Checkable `QPushButton` | Secondary, reversible, no silent overwrites |
| `PrimaryActionButton` | `QPushButton` | Solid accent, visible focus, correct disabled/busy state |
| `PreviewPanel` | `QFrame`, audio controls, labels | Empty until real sample; no fake timing or waveform |
| `AudioTransport` | `QPushButton`, duration/progress label | Play/Pause/Stop based on real audio state |
| `ProgressStatus` | `QLabel`, `QProgressBar` | Differentiate downloading/loading/generating |
| `AdvancedSettings` | `QToolButton`/`QGroupBox` + form controls | Collapsed by default; reflect existing backend capability |
| `InlineError` | `QLabel`, optional retry control | Plain language, associated with the affected task |

## Visual variants

### Buttons

- `primary`: solid accent background, readable foreground, only one prominent instance per screen state.
- `secondary`: neutral surface or outline; used for Save, Download, Regenerate where appropriate.
- `quiet`: minimal chrome for advanced disclosure, help and tertiary actions.
- `disabled`: clearly noninteractive with an explanation when helpful.
- `busy`: prevent repeat submission and show meaningful state (not just spinning forever).
- `focus-visible`: contrasting outline independent of hover and brand fill.

### Inputs

- Default neutral border, hover subtle, focus clearly visible, disabled distinct.
- Empty input includes a visible field label and example placeholder.
- Errors render adjacent to field, preferably with programmatic association.
- Preserve content when moving between loading and error states.

### Navigation

- Sidebar navigation should appear like navigation, not a set of primary buttons.
- Selected state: subtle neutral background + accent bar/icon/text, not an orange-fill.
- Hover does not look identical to selected.
- Keyboard selection and focus are differentiated.

### Audio preview

- **Empty:** short informative placeholder; playback and save controls absent or disabled with explanation.
- **Ready:** enable real playback, show real duration, accessible labels, contextual Save/Download.
- **Playing:** Play button becomes Pause, update status appropriately.
- **Failed playback:** preserve output path and offer recovery without rerunning synthesis unnecessarily.

## PySide6 implementation guidance

- Centralize theme and dimensions in a small module or helper; use QSS for flat colors, borders, padding and selected/focus states.
- Use `objectName` or dynamic properties for style variants; avoid broad selectors that accidentally restyle all widgets.
- Prefer Qt `QLayout` composition; avoid pixel-based absolute positions.
- Use `QSizePolicy`, layout stretches, sensible content margins, and window resize tests.
- For native behavior (keyboard, focus, accessibility) prefer standard Qt widgets over custom-painted imitations.
- Keep widget creation, presentation state and synthesis execution separate where feasible. Avoid a large rewrite in the first PR.
- Main UI entry points are currently `qt_app.py` and `qt_audiobook.py`; consult the existing code before creating or relocating classes.
- Do not change `model_loader.py` or audio persistence as part of a purely visual component PR.

## Suggested theme identifiers (illustrative, not API)

```python
# Conceptual example only; adapt to the actual widget architecture.
COLORS = {
    "canvas": "#101114",
    "sidebar": "#15171B",
    "surface": "#1B1D22",
    "border": "#383B43",
    "text": "#F5F5F5",
    "text_secondary": "#B7BBC4",
    "accent": "#F25C07",
}
SPACING = (4, 8, 12, 16, 24, 32)
RADII = {"control": 6, "surface": 10}
```

## Anti-patterns

- Do not place two bright orange surfaces for both the selected nav item and the primary action.
- Do not show flash-attention or dtype options as mandatory decisions for beginners.
- Do not present a ready preview, waveform, duration, or sample success before data exists.
- Do not hide controls needed to complete an existing supported user task in an inaccessible menu.
- Do not introduce decorative gradients, glassmorphism, or skeuomorphic materials.
- Do not change model defaults invisibly to make screenshots appear ready.

## Open component decisions

- Whether model selection belongs in main flow or only advanced settings.
- Whether a compact sample-text expander is necessary on Create Voice.
- Whether waveforms are feasible with the current playback data or should be replaced by a simple progress line.
- Whether suggestions remain in first release (they are optional).
