#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
GEN_PY="${SCRIPT_DIR}/zimage_2cascade_inference.py"

# -----------------------------
# Editable parameters
# -----------------------------
PYTHON_BIN="${PYTHON_BIN:-python}"
USE_ACCELERATE="${USE_ACCELERATE:-1}"   # 1: accelerate launch, 0: plain python
NUM_PROCESSES="${NUM_PROCESSES:-8}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29502}"

TEST_JSON="${TEST_JSON:-/local/liyanting/datasets/PortraitCraft/track_2_test.json}"
PREDICTED_SIZES_JSON="${PREDICTED_SIZES_JSON:-/local/liyanting/portrait/last_version/track_2_test_with_predicted_sizes.json}"
OUTPUT_PATH="${OUTPUT_PATH:-/local/liyanting/portrait/cvpr2026_challenge_portraitcraft/outputs_zimage_2cascade_001}"

MODEL_PATH="${MODEL_PATH:-/local/liyanting/checkpoints/Tongyi-MAI-Z-Image}"
IMG2IMG_MODEL_PATH="${IMG2IMG_MODEL_PATH:-${MODEL_PATH}}"
USE_LORA_GEN="${USE_LORA_GEN:-1}"
USE_LORA_IMG2IMG="${USE_LORA_IMG2IMG:-0}"
LORA_PATH="${LORA_PATH:-/local/liyanting/checkpoints/zimage_lora/lora_zimage_out_merge_3_joint_v2/final}"
IMG2IMG_LORA_PATH="${IMG2IMG_LORA_PATH:-${LORA_PATH}}"
LORA_SCALE="${LORA_SCALE:-1.0}"

REFERENCE_IMAGE_DIR="${REFERENCE_IMAGE_DIR:-}"

STEPS="${STEPS:-50}"
IMG2IMG_STEPS="${IMG2IMG_STEPS:-50}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-4.0}"
IMG2IMG_GUIDANCE_SCALE="${IMG2IMG_GUIDANCE_SCALE:-4.0}"
STRENGTH="${STRENGTH:-0.99}"
MAX_SEQUENCE_LENGTH="${MAX_SEQUENCE_LENGTH:-768}"
SEED="${SEED:-42}"

WIDTH="${WIDTH:-1024}"                   # fallback only if missing prediction
HEIGHT="${HEIGHT:-1024}"                 # fallback only if missing prediction
AUTO_RESOLUTION="${AUTO_RESOLUTION:-0}" # fallback only if missing prediction
USE_NEGATIVE_PROMPT="${USE_NEGATIVE_PROMPT:-1}"
CFG_NORMALIZATION="${CFG_NORMALIZATION:-0}"
SAVE_PROMPTS="${SAVE_PROMPTS:-1}"

BATCH_START="${BATCH_START:-0}"
BATCH_END="${BATCH_END:--1}"

ARGS=(
  --test_json "${TEST_JSON}"
  --predicted_sizes_json "${PREDICTED_SIZES_JSON}"
  --output_path "${OUTPUT_PATH}"
  --model_path "${MODEL_PATH}"
  --img2img_model_path "${IMG2IMG_MODEL_PATH}"
  --use_lora_gen "${USE_LORA_GEN}"
  --use_lora_img2img "${USE_LORA_IMG2IMG}"
  --lora_path "${LORA_PATH}"
  --img2img_lora_path "${IMG2IMG_LORA_PATH}"
  --lora_scale "${LORA_SCALE}"
  --reference_image_dir "${REFERENCE_IMAGE_DIR}"
  --steps "${STEPS}"
  --img2img_steps "${IMG2IMG_STEPS}"
  --guidance_scale "${GUIDANCE_SCALE}"
  --img2img_guidance_scale "${IMG2IMG_GUIDANCE_SCALE}"
  --strength "${STRENGTH}"
  --max_sequence_length "${MAX_SEQUENCE_LENGTH}"
  --seed "${SEED}"
  --width "${WIDTH}"
  --height "${HEIGHT}"
  --batch_start "${BATCH_START}"
  --batch_end "${BATCH_END}"
)

if [[ "${AUTO_RESOLUTION}" == "1" ]]; then
  ARGS+=(--auto_resolution)
fi
if [[ "${USE_NEGATIVE_PROMPT}" == "1" ]]; then
  ARGS+=(--use_negative_prompt)
fi
if [[ "${CFG_NORMALIZATION}" == "1" ]]; then
  ARGS+=(--cfg_normalization)
fi
if [[ "${SAVE_PROMPTS}" == "1" ]]; then
  ARGS+=(--save_prompts)
fi

echo "Running:"
if [[ "${USE_ACCELERATE}" == "1" ]]; then
  printf ' %q' accelerate launch --num_processes "${NUM_PROCESSES}" --main_process_port "${MAIN_PROCESS_PORT}" -- "${GEN_PY}" "${ARGS[@]}"
  echo
  exec accelerate launch --num_processes "${NUM_PROCESSES}" --main_process_port "${MAIN_PROCESS_PORT}" -- "${GEN_PY}" "${ARGS[@]}"
else
  printf ' %q' "${PYTHON_BIN}" "${GEN_PY}" "${ARGS[@]}"
  echo
  exec "${PYTHON_BIN}" "${GEN_PY}" "${ARGS[@]}"
fi
