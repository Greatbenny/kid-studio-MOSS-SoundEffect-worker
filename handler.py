import base64
import gc
import hashlib
import io
import os
import threading
import time
import wave
from typing import Any, Dict

import runpod
import torch

MODEL_ID = os.getenv("MODEL_ID", "OpenMOSS-Team/MOSS-SoundEffect-v2.0")
MODEL_REVISION = os.getenv("MODEL_REVISION") or None
HF_CACHE = os.getenv("HF_HUB_CACHE", "/runpod-volume/huggingface/hub")
GPU_PRICE_PER_HOUR = os.getenv("RUNPOD_GPU_PRICE_PER_HOUR")
MAX_OUTPUT_BYTES = 9_000_000

_PIPE = None
_LOAD_LOCK = threading.Lock()
_INFERENCE_LOCK = threading.Lock()

QUALITY_PRESETS = {
    "cheap": {"steps": 40, "cfg_scale": 3.5},
    "balanced": {"steps": 70, "cfg_scale": 4.0},
    "high": {"steps": 100, "cfg_scale": 4.0},
}


def _gpu() -> Dict[str, Any]:
    if not torch.cuda.is_available():
        return {"available": False}
    free, total = torch.cuda.mem_get_info()
    return {
        "available": True,
        "name": torch.cuda.get_device_name(0),
        "free_vram_bytes": int(free),
        "total_vram_bytes": int(total),
    }


def _load():
    global _PIPE
    if _PIPE is not None:
        return _PIPE
    with _LOAD_LOCK:
        if _PIPE is None:
            if not torch.cuda.is_available():
                raise RuntimeError("A CUDA GPU is required.")
            from moss_soundeffect_v2 import MossSoundEffectPipeline

            kwargs = {
                "torch_dtype": torch.bfloat16,
                "device": "cuda",
                "cache_dir": HF_CACHE,
            }
            if MODEL_REVISION:
                kwargs["revision"] = MODEL_REVISION
            _PIPE = MossSoundEffectPipeline.from_pretrained(MODEL_ID, **kwargs)
    return _PIPE


def _unload() -> None:
    global _PIPE
    with _LOAD_LOCK:
        _PIPE = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _wav_bytes(audio: torch.Tensor, sample_rate: int) -> bytes:
    wav = audio.detach().float().cpu()
    if wav.ndim == 3:
        wav = wav[0]
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    wav = wav.clamp(-1, 1)
    pcm = (wav * 32767.0).round().to(torch.int16)
    interleaved = pcm.transpose(0, 1).contiguous().numpy().tobytes()
    out = io.BytesIO()
    with wave.open(out, "wb") as fh:
        fh.setnchannels(int(pcm.shape[0]))
        fh.setsampwidth(2)
        fh.setframerate(int(sample_rate))
        fh.writeframes(interleaved)
    return out.getvalue()


def _duration(value: Any, kind: str) -> float:
    default = 15.0 if kind == "ambience" else 5.0
    seconds = round(float(value if value is not None else default), 1)
    if seconds < 1.0 or seconds > 30.0:
        raise ValueError("duration_seconds must be between 1 and 30.")
    return seconds


def _estimated_charge(seconds: float):
    if not GPU_PRICE_PER_HOUR:
        return None
    try:
        return round(seconds * float(GPU_PRICE_PER_HOUR) / 3600.0, 8)
    except ValueError:
        return None


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    inp = job.get("input") or {}
    operation = str(inp.get("operation") or "generate").lower()

    if operation in {"health", "preflight"}:
        return {
            "ok": True,
            "operation": operation,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "model_loaded": _PIPE is not None,
            "gpu": _gpu(),
            "limits": {"max_duration_seconds": 30, "sample_rate": 48000},
            "quality_presets": QUALITY_PRESETS,
        }
    if operation == "warmup":
        started = time.perf_counter()
        pipe = _load()
        return {
            "ok": True,
            "model_loaded": True,
            "model_id": MODEL_ID,
            "sample_rate": pipe.sample_rate,
            "load_seconds": round(time.perf_counter() - started, 3),
            "gpu": _gpu(),
        }
    if operation == "unload":
        _unload()
        return {"ok": True, "model_loaded": False}

    prompt = str(inp.get("prompt") or "").strip()
    if not prompt:
        raise ValueError("prompt is required.")
    if len(prompt) > 2000:
        raise ValueError("prompt must be 2,000 characters or fewer.")

    kind = str(inp.get("kind") or "sfx").lower()
    if kind not in {"sfx", "ambience"}:
        raise ValueError("kind must be sfx or ambience.")

    quality = str(inp.get("quality") or "cheap").lower()
    if quality not in QUALITY_PRESETS:
        raise ValueError("quality must be cheap, balanced, or high.")
    preset = QUALITY_PRESETS[quality]
    steps = int(inp.get("num_inference_steps") or preset["steps"])
    if steps < 10 or steps > 150:
        raise ValueError("num_inference_steps must be between 10 and 150.")

    cfg_scale = float(inp.get("cfg_scale") or preset["cfg_scale"])
    seconds = _duration(inp.get("duration_seconds"), kind)
    seed = int(inp.get("seed", 0))
    negative_prompt = str(
        inp.get("negative_prompt")
        or ("speech, dialogue, vocals, music" if kind == "ambience" else "speech, dialogue, vocals")
    ).strip()

    pipe = _load()
    started = time.perf_counter()
    with _INFERENCE_LOCK:
        audio = pipe(
            prompt=prompt,
            seconds=seconds,
            num_inference_steps=steps,
            cfg_scale=cfg_scale,
            seed=seed,
            negative_prompt=negative_prompt,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - started

    raw = _wav_bytes(audio, pipe.sample_rate)
    if len(raw) > MAX_OUTPUT_BYTES:
        raise RuntimeError("Generated audio is too large for an inline RunPod response.")

    return {
        "ok": True,
        "operation": "generate",
        "kind": kind,
        "audio_base64": base64.b64encode(raw).decode("ascii"),
        "mime_type": "audio/wav",
        "file_extension": "wav",
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "duration_seconds": seconds,
        "sample_rate": int(pipe.sample_rate),
        "channels": int(audio.shape[1] if audio.ndim == 3 else 1),
        "seed": seed,
        "quality": quality,
        "num_inference_steps": steps,
        "cfg_scale": cfg_scale,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "upstream_license": "Apache-2.0",
        "inference_seconds": round(inference_seconds, 3),
        "billing_inputs": {
            "gpu_execution_seconds": round(inference_seconds, 3),
            "configured_gpu_price_per_hour_usd": (
                float(GPU_PRICE_PER_HOUR) if GPU_PRICE_PER_HOUR else None
            ),
            "calculated_execution_charge_usd": _estimated_charge(inference_seconds),
            "note": "Use the RunPod job/API billing record as the authoritative provider charge.",
        },
        "gpu": _gpu(),
    }


runpod.serverless.start({"handler": handler})
