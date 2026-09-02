# Kid Studio MOSS-SoundEffect v2 worker

A RunPod Serverless queue worker for text-to-sound-effect and text-to-ambience generation using the official [MOSS-SoundEffect v2.0](https://huggingface.co/OpenMOSS-Team/MOSS-SoundEffect-v2.0) pipeline.

## What it provides

- Short sound effects and longer ambience beds
- 48 kHz mono WAV output, returned as base64
- Cheap generation by default; balanced and high presets only when requested
- Durable Hugging Face and Torch caches on a RunPod network volume
- Health, preflight, warmup, unload, and generate operations
- Execution metadata for joining results to RunPod's authoritative billing record
- Apache-2.0 upstream code/model license metadata

The image pins the official MOSS-TTS source to commit `c0880299e8b8d0f7119efab17e4e776fffe7b8fa`. MOSS-SoundEffect v2 needs its own Python 3.12/Torch 2.9 CUDA 12.8 environment; it must not be installed into the older top-level MOSS-TTS environment.

## RunPod deployment

In RunPod choose **Serverless → Deploy → Deploy from a GitHub repository**.

| Setting | Value |
|---|---|
| Repository | `Greatbenny/kid-studio-MOSS-SoundEffect-worker` |
| Branch | `main` |
| Dockerfile path | `/Dockerfile` |
| Endpoint type | Queue |
| GPU | 24 GB recommended |
| Workers | Minimum 0, maximum 1 initially |
| Network volume mount | `/runpod-volume` |
| Container disk | 20 GB or more |
| Volume | 50 GB recommended |

The first request downloads the weights and may take several minutes. Keep the shared network volume attached so later workers reuse the model cache.

Recommended environment variables:

- `MODEL_ID=OpenMOSS-Team/MOSS-SoundEffect-v2.0`
- `RUNPOD_GPU_PRICE_PER_HOUR=<the exact rate shown for your selected GPU>`
- `HF_TOKEN=<optional, only if Hugging Face requires it>`
- `TORCHDYNAMO_DISABLE=1` (already set in the image for reliability)

## Request

```json
{
  "input": {
    "operation": "generate",
    "kind": "ambience",
    "prompt": "Gentle morning forest ambience with soft birds and distant leaves rustling, no music or speech.",
    "duration_seconds": 10,
    "quality": "cheap",
    "seed": 42
  }
}
```

`kind` is `sfx` or `ambience`. The duration range is 1–30 seconds.

Quality presets:

| Quality | Steps | Intended use |
|---|---:|---|
| cheap | 40 | Default previews and ordinary assets |
| balanced | 70 | Explicit quality upgrade |
| high | 100 | Final/high-priority assets |

The request can override `num_inference_steps` (10–150), `cfg_scale`, and `negative_prompt`.

## Operations

- `health` or `preflight`: GPU/model status without loading weights
- `warmup`: download/load weights
- `unload`: release GPU memory
- `generate`: create audio

## Billing

The worker returns measured GPU inference time. If `RUNPOD_GPU_PRICE_PER_HOUR` is configured, it also calculates inference-time cost at that rate. This is useful telemetry, but RunPod's job/billing API remains the authoritative actual provider charge because platform billing can include startup and other metered time.

## Safety and production notes

- Review generated audio before publishing.
- Confirm the model card and license remain suitable for the intended commercial use.
- Do not assume generated audio is automatically free of third-party rights concerns.
- Start with minimum workers 0 to avoid idle GPU cost.
