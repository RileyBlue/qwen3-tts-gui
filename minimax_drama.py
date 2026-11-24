#!/usr/bin/env python3
"""
minimax_drama.py
Source text -> segment plan (local LLM) -> dialogue fail-safe -> reference images
-> MiniMax H3 ref2va clips (ComfyUI) -> final stitched vertical video.

Exit codes: 0 ok | 2 [DIALOGUE MISSING] | 3 plan/validation error | 4 ComfyUI/ffmpeg error
"""
import argparse
import copy
import hashlib
import json
import math
import random
import re
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

import requests

# ============================== CONFIG ==============================
COMFY_URL = "http://127.0.0.1:8188"

# Any OpenAI-compatible local endpoint (Ollama, LM Studio, llama.cpp server, vLLM)
LLM_BASE_URL = "http://127.0.0.1:11434/v1"
LLM_MODEL = "qwen2.5:32b-instruct"
LLM_API_KEY = "local"
LLM_TEMPERATURE = 0.3

BASE_DIR = Path(__file__).resolve().parent
REFERENCES_DIR = BASE_DIR / "references"          # change to Path("/references") if needed
WORKFLOW_DIR = BASE_DIR / "workflows"
VIDEO_WORKFLOW = WORKFLOW_DIR / "video_minimax_h3_r2v_api.json"
IMAGE_WORKFLOW = WORKFLOW_DIR / "image-Gen_api.json"
TEMPLATE_FILE = BASE_DIR / "prompt_template.md"
PROJECTS_DIR = BASE_DIR / "projects"
REGISTRY_FILE = REFERENCES_DIR / "characters.json"

MIN_DURATION = 4
MAX_DURATION = 10
MAX_REFS = 3
WORDS_PER_SECOND = 2.5
DIALOGUE_BUFFER_S = 1.0
SPEAKER_SWITCH_S = 0.5
FPS = 24

# --- Video workflow node ids (MiniMax H3) ---
V_PROMPT = "138"
V_DURATION = "132"
V_SEED = "129"
V_LIGHTNING = "146"
V_SAVE = "147"
V_R2V = "136"
V_DEFAULT_LOADERS = ["137", "139"]
V_REF_NODE_BASE = 900

# --- Image workflow node ids (Illustrious) ---
I_POSITIVE = "20:3"
I_KSAMPLER = "20:5"
I_DETAILER = "39:38"
I_OUTPUT = "14"          # PreviewImage -> replaced with SaveImage

REF_PREFIX = ("masterpiece, best quality, newest, absurdres, very aesthetic, high resolution, "
              "vibrant colors, crisp cel shading, flat color palette, clean lineart, full body, "
              "single character,")

STYLE_WITH_CHARS = ("2D flat anime-style animation, clean thin colored lineart, soft minimal cel "
                    "shading, {palette}, large expressive glossy eyes, soft rosy blush, smooth simple "
                    "shapes, flat illustrated backgrounds with simplified detail, cozy slice-of-life "
                    "cartoon aesthetic{tone}, consistent with the character reference art style. "
                    "Vertical 9:16.")
STYLE_NO_CHARS = ("2D flat anime-style animation, clean thin colored lineart, soft minimal cel "
                  "shading, {palette}, smooth simple shapes, light airy colors, flat illustrated "
                  "backgrounds with simplified detail, cozy slice-of-life cartoon aesthetic{tone}. "
                  "Vertical 9:16.")

DIALOGUE_MISSING = "[DIALOGUE MISSING]"
VIDEO_EXTS = (".mp4", ".webm", ".mov", ".mkv")
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")


class PlanError(Exception):
    pass


class DialogueMissingError(PlanError):
    """The source requires dialogue that the writer did not provide."""


# ============================== HELPERS ==============================
def slug(s, n=40):
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")[:n] or "segment"


def norm(s):
    """Normalize text for verbatim dialogue matching (quotes, dashes, case, punctuation)."""
    s = str(s).lower().replace("\u2019", "'").replace("\u2018", "'")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(s.split())


def strip_think(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def dialogue_missing_exit(lines):
    print(DIALOGUE_MISSING)
    for line in lines:
        print(f"- {line}")
    sys.exit(2)


# ============================== LLM PLANNING ==============================
def build_system_prompt(registry):
    t = TEMPLATE_FILE.read_text(encoding="utf-8")
    known = json.dumps(list(registry.values()), indent=2, ensure_ascii=False) if registry else "(none yet)"
    for k, v in {
        "{{MAX_DURATION}}": str(MAX_DURATION),
        "{{MIN_DURATION}}": str(MIN_DURATION),
        "{{MAX_REFS}}": str(MAX_REFS),
        "{{WPS}}": str(WORDS_PER_SECOND),
        "{{KNOWN_CHARACTERS}}": known,
    }.items():
        t = t.replace(k, v)
    return t


def call_llm(system_prompt, source, base_url=None, api_key=None, model=None, temperature=None):
    r = requests.post(
        f"{(base_url or LLM_BASE_URL).rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {api_key or LLM_API_KEY}"},
        json={
            "model": model or LLM_MODEL,
            "temperature": LLM_TEMPERATURE if temperature is None else temperature,
            "temperature": LLM_TEMPERATURE,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"<source>\n{source}\n</source>"},
            ],
        },
        timeout=1800,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def check_failsafe_text(text):
    """Fail-safe layer 1: the LLM itself reported missing dialogue."""
    cleaned = strip_think(text)
    if DIALOGUE_MISSING in cleaned:
        raise DialogueMissingError(cleaned[cleaned.index(DIALOGUE_MISSING):].strip())


def extract_json(text):
    t = strip_think(text)
    t = re.sub(r"^```(?:json)?\s*$", "", t, flags=re.M).replace("```", "")
    s, e = t.find("{"), t.rfind("}")
    if s < 0 or e < 0:
        raise PlanError("LLM response contained no JSON object.")
    return json.loads(t[s:e + 1])


# ============================== VALIDATION ==============================
def validate_plan(plan, source):
    """Fail-safe layer 2 + structural checks. Mutates durations in place."""
    nsrc = f" {norm(source)} "
    errors, warnings, invented = [], [], []

    chars = {c["id"]: c for c in plan.get("characters", [])}
    segs = sorted(plan.get("segments", []), key=lambda s: int(s["segment"]))
    if not segs:
        raise PlanError("Plan has no segments.")
    nums = [int(s["segment"]) for s in segs]
    if nums != list(range(1, len(segs) + 1)):
        errors.append(f"Segments must be numbered 1..{len(segs)} with no gaps; got {nums}")

    for c in chars.values():
        if c.get("gender", "").lower() not in ("male", "female"):
            errors.append(f"Character '{c['id']}' gender must be 'male' or 'female'.")

    for s in segs:
        n = int(s["segment"])
        body = s.get("prompt", "")
        dlg = s.get("dialogue", []) or []
        refs = s.get("refs", []) or []

        # --- verbatim dialogue check ---
        for d in dlg:
            t = norm(d.get("text", ""))
            if not t:
                continue
            if f" {t} " not in nsrc:
                invented.append(f'Segment {n}: "{d["text"]}" does not appear in the source.')
            if t not in norm(body):
                warnings.append(f"Segment {n}: dialogue line not found verbatim in prompt body: \"{d['text']}\"")
            if d.get("speaker") not in chars:
                warnings.append(f"Segment {n}: speaker '{d.get('speaker')}' is not in characters list.")
        if re.search(r"\binvented\b", body, re.I):
            invented.append(f"Segment {n}: prompt body is marked as invented.")

        # --- reference slots ---
        if len(refs) > MAX_REFS:
            errors.append(f"Segment {n}: {len(refs)} refs exceeds MAX_REFS={MAX_REFS}.")
        for r in refs:
            if r not in chars:
                errors.append(f"Segment {n}: ref '{r}' is not defined in characters.")
            elif not chars[r].get("needs_reference", True):
                errors.append(f"Segment {n}: ref '{r}' is marked needs_reference=false.")
        used = {int(x) for x in re.findall(r"<Image\s+(\d+)>", body)}
        bad = [u for u in used if u < 1 or u > len(refs)]
        if bad:
            errors.append(f"Segment {n}: uses <Image {bad}> but only {len(refs)} refs are assigned.")
        for i in range(1, len(refs) + 1):
            if i not in used:
                warnings.append(f"Segment {n}: <Image {i}> ({refs[i-1]}) is assigned but never mentioned.")

        # --- duration vs dialogue ---
        need = 0.0
        if dlg:
            words = sum(len(d.get("text", "").split()) for d in dlg)
            switches = sum(1 for i in range(1, len(dlg)) if dlg[i].get("speaker") != dlg[i - 1].get("speaker"))
            need = words / WORDS_PER_SECOND + DIALOGUE_BUFFER_S + SPEAKER_SWITCH_S * switches
        if need > MAX_DURATION:
            errors.append(f"Segment {n}: dialogue needs ~{need:.1f}s (> {MAX_DURATION}s). Split this segment.")
        dur = float(s.get("duration", MIN_DURATION))
        new_dur = min(MAX_DURATION, max(dur, MIN_DURATION, math.ceil(need)))
        if new_dur != dur:
            warnings.append(f"Segment {n}: duration adjusted {dur}s -> {new_dur}s.")
        s["duration"] = new_dur

    if invented:
        raise DialogueMissingError(DIALOGUE_MISSING + "\n" + "\n".join(f"- {line}" for line in invented))
    for w in warnings:
        print(f"  [warn] {w}")
    if errors:
        raise PlanError("\n".join(errors))
    plan["segments"] = segs
    return plan


# ============================== COMFYUI CLIENT ==============================
class Comfy:
    def __init__(self, url):
        self.url = url.rstrip("/")
        self.client_id = str(uuid.uuid4())

    def upload_image(self, path):
        with open(path, "rb") as f:
            r = requests.post(f"{self.url}/upload/image",
                              files={"image": (path.name, f, "image/png")},
                              data={"overwrite": "true", "type": "input"})
        r.raise_for_status()
        j = r.json()
        return f"{j['subfolder']}/{j['name']}" if j.get("subfolder") else j["name"]

    def queue(self, workflow):
        r = requests.post(f"{self.url}/prompt", json={"prompt": workflow, "client_id": self.client_id})
        if r.status_code != 200:
            raise RuntimeError(f"ComfyUI rejected prompt: {r.text}")
        j = r.json()
        if j.get("node_errors"):
            raise RuntimeError(f"Node errors: {json.dumps(j['node_errors'], indent=2)}")
        return j["prompt_id"]

    def wait(self, prompt_id, timeout=7200, poll=3):
        start = time.time()
        while True:
            r = requests.get(f"{self.url}/history/{prompt_id}")
            r.raise_for_status()
            hist = r.json()
            if prompt_id in hist:
                entry = hist[prompt_id]
                status = entry.get("status", {})
                if status.get("status_str") == "error":
                    raise RuntimeError(f"Execution error: {json.dumps(status.get('messages', []))[:3000]}")
                if status.get("completed", True):
                    return entry
            if time.time() - start > timeout:
                raise TimeoutError(f"Timed out waiting for {prompt_id}")
            time.sleep(poll)

    @staticmethod
    def output_files(entry, node_id):
        files = []
        for v in entry.get("outputs", {}).get(node_id, {}).values():
            if isinstance(v, list):
                files += [i for i in v if isinstance(i, dict) and "filename" in i]
        return files

    def download(self, item, dest):
        dest.parent.mkdir(parents=True, exist_ok=True)
        r = requests.get(f"{self.url}/view", stream=True, params={
            "filename": item["filename"],
            "subfolder": item.get("subfolder", ""),
            "type": item.get("type", "output"),
        })
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)


def project_dir(project):
    return PROJECTS_DIR / slug(project)


def project_references_dir(project):
    return project_dir(project) / "references"


# ============================== REFERENCES ==============================
def _refs_root(refs_dir=None):
    return Path(refs_dir) if refs_dir is not None else REFERENCES_DIR


def load_registry(refs_dir=None):
    reg_file = _refs_root(refs_dir) / "characters.json"
    return load_json(reg_file) if reg_file.exists() else {}


def merge_registry(registry, plan, refs_dir=None):
    keep = ("id", "name", "gender", "needs_reference", "appearance", "ref_tags", "voice",
            "voice_name", "reference")
    for c in plan["characters"]:
        if c["id"] not in registry:
            registry[c["id"]] = {k: c.get(k) for k in keep}
    save_json(_refs_root(refs_dir) / "characters.json", registry)
    return registry


def find_reference(char_id, char=None, refs_dir=None):
    assigned = Path((char or {}).get("reference", ""))
    if assigned.is_file():
        return assigned
    root = _refs_root(refs_dir)
    for stem in (f"{char_id}_ref", char_id):
        for ext in IMAGE_EXTS:
            p = root / f"{stem}{ext}"
            if p.exists():
                return p
    return None


def generate_reference(comfy, base_wf, char, seed=None, refs_dir=None):
    root = _refs_root(refs_dir)
    root.mkdir(parents=True, exist_ok=True)
    wf = copy.deepcopy(base_wf)
    gender = char["gender"].strip().lower()
    wf[I_POSITIVE]["inputs"]["text"] = f"{REF_PREFIX}\n\n{gender}, {char['ref_tags']}"
    if seed is not None:
        wf[I_KSAMPLER]["inputs"]["seed"] = seed
        wf[I_DETAILER]["inputs"]["seed"] = seed
    wf[I_OUTPUT] = {
        "class_type": "SaveImage",
        "inputs": {"images": [I_DETAILER, 0], "filename_prefix": f"drama_refs/{char['id']}_ref"},
    }
    print(f"  Generating reference for '{char['id']}' ({gender})...")
    entry = comfy.wait(comfy.queue(wf))
    files = comfy.output_files(entry, I_OUTPUT)
    if not files:
        raise RuntimeError(f"No image returned for {char['id']}")
    dest = root / f"{char['id']}_ref.png"
    comfy.download(files[0], dest)
    print(f"  Saved {dest}")
    return dest


def ensure_references(comfy, plan, ref_seed=None, refs_dir=None, registry=None):
    root = _refs_root(refs_dir)
    root.mkdir(parents=True, exist_ok=True)
    reg = registry if registry is not None else load_registry(root)
    chars = {c["id"]: c for c in plan["characters"]}
    needed = sorted({r for s in plan["segments"] for r in s.get("refs", [])})
    paths, generated = {}, []
    img_wf = None
    for cid in needed:
        char = {**reg.get(cid, {}), **chars[cid]}
        p = find_reference(cid, char, refs_dir=root)
        if p is None:
            img_wf = img_wf or load_json(IMAGE_WORKFLOW)
            p = generate_reference(comfy, img_wf, chars[cid], ref_seed, refs_dir=root)
            generated.append(cid)
        else:
            print(f"  Found reference for '{cid}': {p.name}")
        paths[cid] = p
    return paths, generated


# ============================== VIDEO ==============================
def compose_prompt(seg):
    palette = seg.get("palette") or "pastel color palette"
    tone = f", {seg['tone']}" if seg.get("tone") else ""
    head = (STYLE_WITH_CHARS if seg.get("refs") else STYLE_NO_CHARS).format(palette=palette, tone=tone)
    return f"{head}\n\n{seg['prompt'].strip()}"


def build_video_workflow(base_wf, prompt, ref_names, duration, seed, lightning, prefix):
    wf = copy.deepcopy(base_wf)
    wf[V_PROMPT]["inputs"]["value"] = prompt
    wf[V_DURATION]["inputs"]["value"] = float(duration)
    wf[V_SEED]["inputs"]["noise_seed"] = int(seed)
    wf[V_LIGHTNING]["inputs"]["value"] = bool(lightning)
    wf[V_SAVE]["inputs"]["filename_prefix"] = prefix

    r2v = wf[V_R2V]["inputs"]
    for k in [k for k in r2v if k.startswith("ref_images.")]:
        del r2v[k]
    for nid in V_DEFAULT_LOADERS:
        wf.pop(nid, None)
    for i, name in enumerate(ref_names):            # <Image i+1> -> ref_image_i
        nid = str(V_REF_NODE_BASE + i)
        wf[nid] = {"class_type": "LoadImage", "inputs": {"image": name}, "_meta": {"title": f"Ref {i+1}"}}
        r2v[f"ref_images.ref_image_{i}"] = [nid, 0]
    return wf


def render_segments(comfy, plan, uploaded, project, seg_dir, seed, lightning, force):
    base_wf = load_json(VIDEO_WORKFLOW)
    outputs, log = [], []
    seg_dir.mkdir(parents=True, exist_ok=True)
    for seg in plan["segments"]:
        n = int(seg["segment"])
        name = f"seg_{n:03d}_{slug(seg.get('title', 'segment'))}"
        out = seg_dir / f"{name}.mp4"
        if out.exists() and not force:
            print(f"[{n:03d}] exists, skipping ({out.name})")
            outputs.append(out)
            continue
        prompt = compose_prompt(seg)
        (seg_dir / f"{name}.prompt.txt").write_text(prompt, encoding="utf-8")
        s = (seed + n) if seed is not None else random.randint(1, 2**48)
        refs = [uploaded[r] for r in seg.get("refs", [])]
        wf = build_video_workflow(base_wf, prompt, refs, seg["duration"], s, lightning,
                                  f"drama/{project}/{name}")
        print(f"[{n:03d}] rendering '{seg.get('title')}' ({seg['duration']}s, refs={seg.get('refs', [])}, seed={s})")
        entry = comfy.wait(comfy.queue(wf))
        files = comfy.output_files(entry, V_SAVE)
        vids = [f for f in files if Path(f["filename"]).suffix.lower() in VIDEO_EXTS] or files
        if not vids:
            raise RuntimeError(f"Segment {n}: no video output found.")
        comfy.download(vids[0], out)
        print(f"[{n:03d}] saved {out.name}")
        outputs.append(out)
        log.append({"segment": n, "file": out.name, "seed": s, "duration": seg["duration"], "refs": seg.get("refs", [])})
    save_json(seg_dir.parent / "render_log.json", log)
    return outputs


# ============================== CONCAT ==============================
def has_audio(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
                        "stream=index", "-of", "csv=p=0", str(path)], capture_output=True, text=True)
    return bool(r.stdout.strip())


def concat_segments(clips, final_path):
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found on PATH.")
    tmp = final_path.parent / "_normalized"
    tmp.mkdir(parents=True, exist_ok=True)
    normed = []
    for clip in clips:                       # clips arrive in segment order
        dst = tmp / clip.name
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(clip)]
        if has_audio(clip):
            cmd += ["-map", "0:v:0", "-map", "0:a:0"]
        else:
            cmd += ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo", "-map", "0:v:0", "-map", "1:a:0", "-shortest"]
        cmd += ["-vf", f"fps={FPS},format=yuv420p", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                "-c:a", "aac", "-ar", "48000", "-ac", "2", "-b:a", "192k", str(dst)]
        subprocess.run(cmd, check=True)
        normed.append(dst)
    list_file = tmp / "concat.txt"
    list_file.write_text("".join(f"file '{p.resolve().as_posix()}'\n" for p in normed), encoding="utf-8")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(list_file),
                    "-c", "copy", "-movflags", "+faststart", str(final_path)], check=True)
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"\nFinal video: {final_path}")


# ============================== MAIN ==============================
def main():
    ap = argparse.ArgumentParser(description="Source -> MiniMax H3 vertical mini drama")
    ap.add_argument("--source", required=True, help="Path to the writer's source .txt")
    ap.add_argument("--project", help="Project name (default: source filename)")
    ap.add_argument("--plan", help="Use an existing plan JSON instead of calling the LLM")
    ap.add_argument("--replan", action="store_true", help="Ignore cached plan.json and re-run the LLM")
    ap.add_argument("--plan-only", action="store_true", help="Stop after planning and validation")
    ap.add_argument("--stop-after-refs", action="store_true", help="Stop after generating new references for review")
    ap.add_argument("--lightning", action="store_true", help="Enable 4-step turbo LoRA")
    ap.add_argument("--seed", type=int, help="Base video seed (segment seed = seed + segment number)")
    ap.add_argument("--ref-seed", type=int, help="Seed for reference generation (default: workflow's seed)")
    ap.add_argument("--force", action="store_true", help="Re-render segments that already exist")
    args = ap.parse_args()

    source_path = Path(args.source)
    source = source_path.read_text(encoding="utf-8")
    m = re.search(r"<source>(.*?)</source>", source, re.S)
    if m:
        source = m.group(1).strip()
    project = slug(args.project or source_path.stem)
    proj_dir = project_dir(project)
    proj_dir.mkdir(parents=True, exist_ok=True)
    refs_dir = project_references_dir(project)
    refs_dir.mkdir(parents=True, exist_ok=True)
    plan_path = proj_dir / "plan.json"
    src_hash = hashlib.sha256(source.encode()).hexdigest()

    registry = load_registry(refs_dir)

    # ---------- PLAN ----------
    try:
        if args.plan:
            raw = Path(args.plan).read_text(encoding="utf-8")
            check_failsafe_text(raw)
            plan = extract_json(raw)
        elif plan_path.exists() and not args.replan and load_json(plan_path).get("source_hash") == src_hash:
            print(f"Using cached plan: {plan_path}")
            plan = load_json(plan_path)
        else:
            print("Planning segments with LLM...")
            raw = call_llm(build_system_prompt(registry), source)
            (proj_dir / "llm_raw.txt").write_text(raw, encoding="utf-8")
            check_failsafe_text(raw)                  # fail-safe layer 1
            plan = extract_json(raw)
        plan = validate_plan(plan, source)            # fail-safe layer 2
        plan["source_hash"] = src_hash
        save_json(plan_path, plan)
    except DialogueMissingError as e:
        print(str(e))
        sys.exit(2)
    except PlanError as e:
        print(f"PLAN ERROR:\n{e}\nEdit {plan_path} and rerun with --plan {plan_path}")
        sys.exit(3)

    total = sum(s["duration"] for s in plan["segments"])
    print(f"Plan OK: {len(plan['segments'])} segments, ~{total:.0f}s total.")
    for s in plan["segments"]:
        print(f"  {int(s['segment']):03d}  {s['duration']:>4}s  refs={s.get('refs', [])}  {s.get('title', '')}")
    registry = merge_registry(registry, plan, refs_dir)
    if args.plan_only:
        return

    # ---------- REFERENCES ----------
    try:
        comfy = Comfy(COMFY_URL)
        ref_paths, generated = ensure_references(comfy, plan, args.ref_seed,
                                                 refs_dir=refs_dir, registry=registry)
        if generated and args.stop_after_refs:
            print(f"\nGenerated new references: {generated}. Review them in {refs_dir}, then rerun.")
            return
        uploaded = {cid: comfy.upload_image(p) for cid, p in ref_paths.items()}

        # ---------- SEGMENTS ----------
        clips = render_segments(comfy, plan, uploaded, project, proj_dir / "segments",
                                args.seed, args.lightning, args.force)

        # ---------- COMBINE ----------
        concat_segments(clips, proj_dir / f"{project}_final.mp4")
    except (RuntimeError, TimeoutError, requests.RequestException, subprocess.CalledProcessError) as e:
        print(f"ERROR: {e}")
        sys.exit(4)


if __name__ == "__main__":
    main()