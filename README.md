# cvpr2026_challenge_portraitcraft

**CVPR 2026 PortraitCraft Challenge** — a two-stage pipeline: predict canvas size with Qwen3, then cascade **Z-Image** (text-to-image) followed by **Z-Image Turbo** (image-to-image).

## Environment

```bash
git clone https://github.com/ytli-ustc/cvpr2026_challenge_portraitcraft.git
cd cvpr2026_challenge_portraitcraft
pip install -r requirements.txt
```

Python 3.10+ and a CUDA 12.x–compatible PyTorch build are recommended.

## Download checkpoints

Create a `checkpoints` folder at the repository root and download the artifacts below (`huggingface-cli`, `snapshot_download`, or `git lfs` are all fine):

| Purpose | Hugging Face repo | Suggested local path |
|---------|-------------------|----------------------|
| Stage-1 base (txt2img) | [Tongyi-MAI/Z-Image](https://huggingface.co/Tongyi-MAI/Z-Image) | `checkpoints/Z-Image` |
| Stage-2 base (img2img) | [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) | `checkpoints/Z-Image-Turbo` |
| Size predictor base | [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) | `checkpoints/Qwen3-0.6B` |
| Our LoRA + size predictor weights | [ustc-ytli/cvpr26_portraitcraft](https://huggingface.co/ustc-ytli/cvpr26_portraitcraft) | `checkpoints/cvpr26_portraitcraft` |

Example downloads:

```bash
mkdir -p checkpoints

huggingface-cli download Tongyi-MAI/Z-Image \
  --local-dir checkpoints/Z-Image

huggingface-cli download Tongyi-MAI/Z-Image-Turbo \
  --local-dir checkpoints/Z-Image-Turbo

huggingface-cli download Qwen/Qwen3-0.6B \
  --local-dir checkpoints/Qwen3-0.6B

huggingface-cli download ustc-ytli/cvpr26_portraitcraft \
  --local-dir checkpoints/cvpr26_portraitcraft
```

Contents of `ustc-ytli/cvpr26_portraitcraft`:

- `checkpoint-1000/` — Z-Image Stage-1 LoRA  
- `size_predictor/` — Qwen3 aspect-ratio LoRA  

Example layout:

```text
cvpr2026_challenge_portraitcraft/
├── assets/
│   ├── buckets.json
│   └── test_buckets.jsonl
├── checkpoints/
│   ├── Z-Image/
│   ├── Z-Image-Turbo/
│   ├── Qwen3-0.6B/
│   └── cvpr26_portraitcraft/
│       ├── checkpoint-1000/
│       └── size_predictor/
├── predict_sizes.py
├── zimage_2cascade_inference.py
└── zimage_2cascade_inference.sh
```

You also need the **official test manifest** (e.g. PortraitCraft Track 2 `track_2_test.json`). It is not shipped in this repo; pass its path via `TEST_JSON` in Step 2.

## Inference

All commands below assume the repository root as the working directory.

### Step 1 — Predict `(width, height)`

Run the size predictor on `assets/test_buckets.jsonl` and write a JSON file with predicted dimensions:

```bash
python predict_sizes.py \
  --base_model checkpoints/Qwen3-0.6B \
  --lora_path checkpoints/cvpr26_portraitcraft/size_predictor \
  --input_jsonl assets/test_buckets.jsonl \
  --output_json track_2_test_with_predicted_sizes.json
```

`--data_dir` is optional; by default `buckets.json` is read from the same directory as `--input_jsonl` (here: `assets/buckets.json`).

### Step 2 — Two-stage Z-Image cascade

Launch `zimage_2cascade_inference.sh`. Parameters are overridden with environment variables. The example matches our recommended setup: Stage-1 = Z-Image + LoRA; Stage-2 = Z-Image-Turbo img2img. Paths below are relative to the repo root except `TEST_JSON`.

```bash
TEST_JSON=path/to/track_2_test.json \
PREDICTED_SIZES_JSON=track_2_test_with_predicted_sizes.json \
OUTPUT_PATH=outputs_zimage_2cascade_001 \
STAGE1_OUTPUT_PATH=outputs_zimage_2cascade_001_stage1 \
MODEL_PATH=checkpoints/Z-Image \
IMG2IMG_MODEL_PATH=checkpoints/Z-Image-Turbo \
LORA_PATH=checkpoints/cvpr26_portraitcraft/checkpoint-1000 \
USE_LORA_GEN=1 \
USE_LORA_IMG2IMG=0 \
STEPS=50 \
GUIDANCE_SCALE=5.0 \
IMG2IMG_STEPS=9 \
IMG2IMG_GUIDANCE_SCALE=0 \
STRENGTH=0.3 \
USE_NEGATIVE_PROMPT_STAGE1=1 \
USE_NEGATIVE_PROMPT_STAGE2=0 \
NUM_PROCESSES=8 \
MAIN_PROCESS_PORT=29502 \
./zimage_2cascade_inference.sh
```

| Variable | Description |
|----------|-------------|
| `TEST_JSON` | Official test prompts (download separately). |
| `PREDICTED_SIZES_JSON` | Output of Step 1: `track_2_test_with_predicted_sizes.json`. |
| `OUTPUT_PATH` | Final images: `outputs_zimage_2cascade_001/`. |
| `STAGE1_OUTPUT_PATH` | Stage-1 intermediates: `outputs_zimage_2cascade_001_stage1/`. |
| `MODEL_PATH` | Stage-1 base: `checkpoints/Z-Image`. |
| `IMG2IMG_MODEL_PATH` | Stage-2 base: `checkpoints/Z-Image-Turbo`. |
| `LORA_PATH` | Stage-1 LoRA: `checkpoints/cvpr26_portraitcraft/checkpoint-1000`. |
| `STRENGTH` | Img2img strength (we use `0.3`). |
| `IMG2IMG_STEPS` / `IMG2IMG_GUIDANCE_SCALE` | Turbo steps and CFG (often `0` CFG for Turbo-style runs). |
| `NUM_PROCESSES` | `accelerate launch` worker count — match your GPU count. |

For single-GPU debugging, set `USE_ACCELERATE=0` or `NUM_PROCESSES=1`.

Final images land under `OUTPUT_PATH`. With `SAVE_PROMPTS=1` (the script default), a `prompts.json` is also emitted.

## File overview

| File | Role |
|------|------|
| `predict_sizes.py` | Qwen3-0.6B + size-predictor LoRA for aspect/size prediction. |
| `zimage_2cascade_inference.py` | Main two-stage inference entry point. |
| `zimage_2cascade_inference.sh` | Multi-GPU launcher (`accelerate launch`). |
| `assets/test_buckets.jsonl` | Sample input lines for size prediction. |
| `assets/buckets.json` | Bucket definitions (letters ↔ resolutions). |

## References

- [Tongyi-MAI/Z-Image](https://huggingface.co/Tongyi-MAI/Z-Image)  
- [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)  
- [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)  
- [ustc-ytli/cvpr26_portraitcraft](https://huggingface.co/ustc-ytli/cvpr26_portraitcraft)  
