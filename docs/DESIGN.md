# Qwen3-TTS Studio — Design contract

**Version:** 0.1 · **Status:** Proposal · **Target:** PySide6 desktop app · **First slice:** Create Voice

> **Design objective:** Help people produce and evaluate their first voice sample with minimal interpretation and minimal setup. Keep professional controls available, but not prominent by default.

This document describes the **target experience**, not the current implementation. Do not claim a component, token, device auto-detection, or feedback state is supported until verified in running software.

## 1. Principles (ordered)

1. **Don't make me think.** Use recognizable labels and explicit outcomes. A first-time user should understand the next action from the screen itself.
2. **One primary action per context.** `Generate voice` is the prominent Create Voice action. Selected navigation is **not** a competing orange CTA.
3. **Progressive disclosure.** Hide inference details (device, dtype, attention, chunking, seed) behind `Advanced settings`; reveal only when useful.
4. **Respect user agency.** Defaults simplify; manual controls and a way to recover remain available. Never silently discard text or a generated result.
5. **Make system status legible.** Distinguish model download, model loading, generation, ready, and failure; give specific recovery choices.
6. **Keep the interface compact, not cramped.** Moderate control heights, strong labels, legible text, intentional whitespace, and meaningful grouping.
7. **Preserve the working product.** Avoid changes to synthesis or storage behavior in a styling-only PR. Refactor incrementally.

## 2. Behavioral design rationale

| Framework | Observed friction / hypothesis | Intervention | Validation |
| --- | --- | --- | --- |
| **B=MAP** | Strong desire to create audio, but low ability due to technical settings and ambiguous actions | A clear description field + one `Generate voice` CTA; ready-to-use defaults | First-sample completion and assistance needed |
| **COM-B** | Psychological capability burden: unfamiliar model concepts | Plain-language fields, optional advanced controls, actionable errors | Fewer misunderstood controls |
| **COM-B** | Physical opportunity burden: scanning a dense screen | Compact sections with predictable hierarchy and a clear result area | Lower navigation/search errors |
| **EAST — Easy** | Setup blocks the goal | Safe defaults, minimal required fields | Fewer pre-generation decisions |
| **EAST — Attractive** | Outcome is hard to anticipate | Clear audio preview, accessible playback, save after success | Preview listening and save success |
| **EAST — Social** | Examples may help discovery but social proof is not required | Optional example descriptions, not fabricated endorsements | Optional preset usefulness |
| **EAST — Timely** | Loading and errors appear without context | Contextual progress and recovery, shown when relevant | Users can explain current state |

These are **design hypotheses**, not measured findings. Do not report numeric usability scores without research.

## 3. Information hierarchy

1. Current location: `Create Voice` (neutral selected navigation)
2. Page heading: `Create a voice`
3. Core input: `Voice description` with a concise example
4. Optional suggestions (may be omitted in v1)
5. Primary action: `Generate voice`
6. Preview region: only an empty state initially; playback/actions appear after success
7. Secondary context: language (if required for current flow), model readiness
8. Advanced settings / troubleshooting

Avoid repeating `Create Voice` in the sidebar, eyebrow, title, icon heading and CTA. The page heading provides context; the CTA is the action.

## 4. Visual foundations (proposed tokens)

**All colors and sizes are starting tokens, not an approved brand system. Verify contrast in context.**

| Token | Value | Intended usage |
| --- | --- | --- |
| `color.bg.canvas` | `#101114` | Application background |
| `color.bg.sidebar` | `#15171B` | Sidebar |
| `color.bg.surface` | `#1B1D22` | Inputs, elevated grouped surfaces |
| `color.bg.surface.hover` | `#252831` | Hovered secondary control |
| `color.border.default` | `#383B43` | Borders and separators |
| `color.text.primary` | `#F5F5F5` | High-emphasis text |
| `color.text.secondary` | `#B7BBC4` | Supporting descriptions |
| `color.text.muted` | `#9399A4` | Metadata (check contrast) |
| `color.accent.primary` | `#F25C07` | Primary action, selected/focused affordances |
| `color.status.success` | `#46BA82` | Ready/success indicators with text |
| `color.status.error` | `#F07878` | Errors, always paired with text |

### Type scale

- Prefer the system UI font via `QFont` first (`SF Pro` on macOS where available; appropriate system fallback elsewhere). Do not bundle proprietary fonts.
- Body: 14 px equivalent; secondary 13 px; labels 13–14 px / semibold.
- Page title: 26–30 px / semibold. Card title: 16–18 px / semibold.
- Small metadata: 12 px only when nonessential; never shrink essential labels to fit.
- Use sensible line spacing and test rendering at Qt scale factors and OS font settings.

### Layout and geometry

- Spacing tokens: `4, 8, 12, 16, 24, 32` px.
- Typical sidebar width: 208–224 px, content max line width ~680 px; resize gracefully.
- Primary button height: 40–44 px. Secondary buttons: 34–38 px. Do not use full-width CTA unless available width is narrow.
- Touch is not the main target; keyboard focus remains essential. Keep click targets comfortably usable (target >= 36 px where practical).
- Radius: 6 px for compact controls, 8–10 px for cards/inputs, 12 px maximum for large sections.
- Use 1 px borders and flat fills. **No gradients, glass effects, skeuomorphism, or oversized shadow effects.**
- Avoid a card for every block. Use spacing and separators for grouping before adding a surface.

## 5. Color and emphasis policy

- **One visually dominant primary CTA:** solid orange `Generate voice`.
- Selected sidebar row: neutral filled surface, subtle accent marker or text/icon treatment; **not an orange-filled button**.
- Input resting state: neutral border. Focus: visible, high-contrast outline/accent independent of hover.
- Presets and secondary actions: neutral outlined or subtle filled; no competing orange fills.
- Status indicators communicate state with icon + text, not color alone.
- Text on orange must meet contrast standards; if white text fails, use dark text or adjust background appropriately based on measured contrast.

## 6. Copy and behavior contracts

| Element | Recommended copy | Rule |
| --- | --- | --- |
| Navigation | `Create Voice` | Location label only |
| Page heading | `Create a voice` | No redundant “with AI” unless it adds meaning |
| Field | `Voice description` | Describe tone/accent/speaking style |
| Placeholder | `e.g. A warm, calm voice with a friendly tone` | Example, not an instructional paragraph |
| CTA | `Generate voice` | Starts generation; disabled only for a real blocker |
| Preview empty | `Your generated sample will appear here.` | No fictitious waveform or playback time |
| Preview ready | `Sample ready` | Expose Play, Save, Download and Regenerate appropriately |
| Model download | `Downloading voice model` | Show real bytes/percent if available; first-use explanation |
| Generation | `Generating sample…` | Busy indicator + cancel only if supported |
| Error | `Couldn't generate the sample` | Include reason in understandable terms and a safe Retry |

**No fabricated timings:** Do not promise a result “in seconds” unless benchmarked across supported hardware. Do not show fake output duration, waveform, progress, or readiness.

## 7. Interaction state contract

- **Idle:** input enabled, `Generate voice` available when inputs valid; preview empty and unobtrusive.
- **Preparing:** differentiate first-time download, local model loading, and runtime check. Indicate what is happening and prevent duplicate submissions.
- **Generating:** show active status; preserve description and selected options; disable incompatible actions only.
- **Success:** preview becomes available with real audio controls; Save and Download are contextual.
- **Error:** preserve inputs and any previously valid output; show cause if known, meaningful retry, and support diagnostic details behind disclosure.
- **Cancel:** only offer if the current generation job can actually be canceled safely; explain consequences.

See [UX-FLOWS.md](UX-FLOWS.md) for transitions and edge cases.

## 8. Accessibility contract

- WCAG 2.2 AA targets for text contrast and interaction states where applicable.
- Every input has a persistent, programmatically associated label; placeholder is never the only label.
- Full keyboard operation, meaningful tab order, visible focus indicators, accessible names for icon-only actions.
- Status and errors exposed to assistive technology through appropriate Qt accessibility interfaces.
- Loading indicators need text; progress values reflect real information.
- Audio controls need accessible Play/Pause states and a real duration once available.
- Avoid using color or waveform alone to indicate playback or selection.
- Support resizing and OS scaling; avoid clipped labels at 125–200% scaling.

## 9. Engineering guardrails

- Use existing **PySide6 / Qt Widgets**. Do not introduce Electron/React for the initial redesign.
- Isolate QSS/theme tokens and shared widget styling where useful; avoid huge string literals duplicated throughout modules.
- Prefer Qt layouts over absolute positioning; use stretch factors, size policies, and minimum sizes carefully.
- Keep inference/loading/filesystem logic in existing non-UI modules. Do not modify model performance in a visual-only change.
- Avoid changing current voice identifiers, output formats, paths, or saved metadata without separate review.
- Device auto-selection (including MPS) is a **separate proposed feature**, not a guarantee of this redesign; do not silently substitute unsupported backends.
- Where internal names remain `VoiceDesign`, user-facing labels may use `Create Voice`, but preserve mappings to existing logic.

## 10. Definition of done (design slice)

A Create Voice UI proposal is ready to implement when: hierarchy and states are defined, no navigation/CTA competition remains, copy matches actions, required inputs are unambiguous, contrast and keyboard checks are planned, and change boundaries are documented. It is only **implemented** once the full checklist in [DESIGN-QA.md](DESIGN-QA.md) passes.

## 11. Decision log

| ID | Decision | Reason | Status |
| --- | --- | --- | --- |
| D-001 | Flat dark mode; no button gradients | Reduce visual noise and preserve hierarchy | Proposed |
| D-002 | Neutral selected sidebar; orange primary CTA | Separate location from action | Proposed |
| D-003 | Compact controls, ample section spacing | Improve scan speed without oversized UI | Proposed |
| D-004 | Technical settings behind disclosure | Support novice and expert use | Proposed |
| D-005 | Real states for download/load/generate | Build trust and remove ambiguous waiting | Proposed |
| D-006 | No automatic MPS claims until implemented | Avoid misleading device status | Required guardrail |

**Maintainer review required** before adoption as an implementation standard.
