"""Model loading utilities for Qwen3-TTS."""

import torch

try:
    from qwen_tts import Qwen3TTSModel
except ImportError:
    Qwen3TTSModel = None


DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def load_model(model_id, device, dtype, attn):
    """Load a Qwen3TTSModel from pretrained weights.

    Args:
        model_id: Hugging Face model ID or local path.
        device: "cuda" or "cpu".
        dtype: "bfloat16", "float16", or "float32".
        attn: Attention implementation ("flash_attention_2", "sdpa", "eager").

    Returns:
        Loaded Qwen3TTSModel instance.

    Raises:
        ImportError: If qwen-tts package is not installed.
        Exception: If model loading fails.
    """
    if Qwen3TTSModel is None:
        raise ImportError(
            "'qwen-tts' package is not installed. "
            "Install it with: pip install -U qwen-tts"
        )

    torch_dtype = DTYPE_MAP[dtype]

    print(f"Loading model: {model_id}")
    print(f"Device: {device} | Dtype: {dtype} | Attention: {attn}")

    model = Qwen3TTSModel.from_pretrained(
        model_id,
        device_map="auto" if device == "cuda" else None,
        dtype=torch_dtype,
        attn_implementation=attn,
        trust_remote_code=True,
    )
    return model


class ModelManager:
    """Lazily loads models by kind and keeps at most `max_loaded` in memory.

    kind is 'voice-design', 'voice-clone', 'xtts', 'csm', 'dia', or 'f5'.
    `settings` is a callable returning (device, dtype, attn); changing them reloads models.
    """

    def __init__(self, settings, max_loaded=2):
        import threading
        from collections import OrderedDict
        self._settings = settings
        self._max = max_loaded
        self._models = OrderedDict()
        self._loaded_with = None
        self._lock = threading.Lock()
        # TTSGenerator uses this to prevent GUI and API generation from sharing a model concurrently.
        self.generation_lock = threading.RLock()

    def get(self, kind, notify=None):
        from config import CSM_MODEL, DIA_MODEL, MODEL_MAP, XTTS_MODEL
        with self._lock:
            current = tuple(self._settings())
            if current != self._loaded_with:
                self._clear_models()
                self._free_memory()
                self._loaded_with = current
            family = "qwen" if kind in MODEL_MAP else kind
            loaded_kind = next(iter(self._models), None)
            loaded_family = "qwen" if loaded_kind in MODEL_MAP else loaded_kind
            if self._models and family != loaded_family:
                self._clear_models()
                self._free_memory()
            if kind in self._models:
                self._models.move_to_end(kind)
                return self._models[kind]
            while len(self._models) >= self._max:
                old_kind, old_model = self._models.popitem(last=False)
                if old_kind == "f5":
                    old_model.close()
                self._free_memory()
            if notify:
                notify(f"Loading {kind} model…")
            if kind == "xtts":
                try:
                    from TTS.api import TTS
                except ImportError as exc:
                    raise ImportError(
                        "XTTS requires the optional coqui-tts package. See the README for installation."
                    ) from exc
                import os
                if os.environ.get("COQUI_TOS_AGREED") != "1":
                    raise ValueError("Accept the XTTS model license before loading XTTS.")
                self._models[kind] = TTS(XTTS_MODEL).to(current[0])
            elif kind == "csm":
                try:
                    from transformers import AutoProcessor, CsmForConditionalGeneration
                except ImportError as exc:
                    raise ImportError("CSM requires Transformers with CSM support (4.52.1 or newer).") from exc
                processor = AutoProcessor.from_pretrained(CSM_MODEL)
                dtype = DTYPE_MAP[current[1]] if current[0] == "cuda" else torch.float32
                model = CsmForConditionalGeneration.from_pretrained(
                    CSM_MODEL, torch_dtype=dtype, attn_implementation="sdpa"
                ).to(current[0])
                self._models[kind] = (processor, model.eval())
            elif kind == "f5":
                from f5_worker import F5Worker
                self._models[kind] = F5Worker(current[0])
            elif kind == "dia":
                try:
                    from transformers import AutoProcessor, DiaForConditionalGeneration
                except ImportError as exc:
                    raise ImportError("Dia requires Transformers with Dia support. See the README.") from exc
                processor = AutoProcessor.from_pretrained(DIA_MODEL)
                processor.audio_tokenizer.to(current[0]).eval()
                dtype = DTYPE_MAP[current[1]] if current[0] == "cuda" else torch.float32
                model = DiaForConditionalGeneration.from_pretrained(
                    DIA_MODEL, torch_dtype=dtype, attn_implementation="sdpa"
                ).to(current[0])
                self._models[kind] = (processor, model.eval())
            else:
                self._models[kind] = load_model(MODEL_MAP[kind], *current)
            return self._models[kind]

    def _clear_models(self):
        worker = self._models.get("f5")
        if worker is not None:
            worker.close()
        self._models.clear()

    @staticmethod
    def _free_memory():
        import gc
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
