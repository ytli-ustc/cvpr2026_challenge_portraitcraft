# cvpr2026_challenge_portraitcraft

CVPR 2026 PortraitCraft Challenge — 两阶段推理方案：先用 Qwen3 预测画幅比例，再用 Z-Image（text2img）+ Z-Image-Turbo（img2img）级联生成。

## 环境准备

```bash
git clone https://github.com/ytli-ustc/cvpr2026_challenge_portraitcraft.git
cd cvpr2026_challenge_portraitcraft
pip install -r requirements.txt
```

建议使用 Python 3.10+ 与 CUDA 12.x 的 PyTorch 环境。

## 下载模型与权重

在仓库根目录创建 `checkpoints` 目录，并从 Hugging Face 拉取以下资源（可用 `huggingface-cli` 或 `git lfs clone`）：

| 用途 | Hugging Face 仓库 | 建议本地路径 |
|------|-------------------|--------------|
| Stage-1 文生图底座 | [Tongyi-MAI/Z-Image](https://huggingface.co/Tongyi-MAI/Z-Image) | `checkpoints/Z-Image` |
| Stage-2 图生图底座 | [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) | `checkpoints/Z-Image-Turbo` |
| 画幅预测底座 | [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) | `checkpoints/Qwen3-0.6B` |
| 本方案 LoRA 与 size predictor | [ustc-ytli/cvpr26_portraitcraft](https://huggingface.co/ustc-ytli/cvpr26_portraitcraft) | `checkpoints/cvpr26_portraitcraft` |

示例下载代码：

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

`ustc-ytli/cvpr26_portraitcraft` 仓库内包含：

- `checkpoints/cvpr26_portraitcraft/checkpoint-1000/` — Z-Image Stage-1 LoRA
- `checkpoints/cvpr26_portraitcraft/size_predictor/` — Qwen3 画幅预测 LoRA

最终目录结构示例：

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

另需准备官方测试集 JSON（例如 PortraitCraft Track 2 的 `track_2_test.json`），在 Step 2 中通过 `TEST_JSON` 指定其路径（该文件不在本仓库内）。

## 推理流程

以下命令均默认在仓库根目录 `cvpr2026_challenge_portraitcraft/` 下执行。

### Step 1：预测画幅（aspect ratio）

对 `assets/test_buckets.jsonl` 中的每条样本预测 `(width, height)`，并写出带尺寸字段的测试 JSON：

```bash
python predict_sizes.py \
  --base_model checkpoints/Qwen3-0.6B \
  --lora_path checkpoints/cvpr26_portraitcraft/size_predictor \
  --input_jsonl assets/test_buckets.jsonl \
  --output_json track_2_test_with_predicted_sizes.json
```

`--data_dir` 可省略，默认使用 `--input_jsonl` 所在目录下的 `buckets.json`（即 `assets/buckets.json`）。

### Step 2：两阶段 Z-Image 级联生成

执行 `zimage_2cascade_inference.sh`。脚本通过环境变量覆盖默认参数；下列示例与推荐推理配置一致（Stage-1：Z-Image + LoRA；Stage-2：Z-Image-Turbo img2img）。除 `TEST_JSON` 外，路径均为仓库内相对路径。

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

说明：

| 变量 | 含义 |
|------|------|
| `TEST_JSON` | 官方测试 prompt 列表（需自行下载，非仓库内文件） |
| `PREDICTED_SIZES_JSON` | Step 1 输出：`track_2_test_with_predicted_sizes.json` |
| `OUTPUT_PATH` | 最终生成图目录：`outputs_zimage_2cascade_001/` |
| `STAGE1_OUTPUT_PATH` | Stage-1 中间结果：`outputs_zimage_2cascade_001_stage1/` |
| `MODEL_PATH` | Stage-1 底座：`checkpoints/Z-Image` |
| `IMG2IMG_MODEL_PATH` | Stage-2 底座：`checkpoints/Z-Image-Turbo` |
| `LORA_PATH` | Stage-1 LoRA：`checkpoints/cvpr26_portraitcraft/checkpoint-1000` |
| `STRENGTH` | img2img 强度（推荐 `0.3`） |
| `IMG2IMG_STEPS` / `IMG2IMG_GUIDANCE_SCALE` | Turbo 阶段步数与 guidance（Turbo 常用 `0`） |
| `NUM_PROCESSES` | `accelerate` 并行进程数，按 GPU 数量调整 |

若仅单卡调试，可设置 `USE_ACCELERATE=0` 或 `NUM_PROCESSES=1`。

生成结果默认保存在 `OUTPUT_PATH`；若开启 `SAVE_PROMPTS=1`（脚本默认），会同时保存对应 prompt 文本。

## 文件说明

| 文件 | 说明 |
|------|------|
| `predict_sizes.py` | 基于 Qwen3-0.6B + size predictor LoRA 预测画幅 |
| `zimage_2cascade_inference.py` | 两阶段推理主程序 |
| `zimage_2cascade_inference.sh` | 多卡启动脚本（`accelerate launch`） |
| `assets/test_buckets.jsonl` | 画幅预测输入样例 |
| `assets/buckets.json` | 画幅类别与分辨率定义 |

## 参考链接

- [Tongyi-MAI/Z-Image](https://huggingface.co/Tongyi-MAI/Z-Image)
- [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)
- [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)
- [ustc-ytli/cvpr26_portraitcraft](https://huggingface.co/ustc-ytli/cvpr26_portraitcraft)
