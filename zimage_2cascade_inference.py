#!/usr/bin/env python3
"""
Two-stage inference based on zimage_lora_zimage_resize.py.

Only key change:
- Stage-1 (text2img) resolution comes from --predicted_sizes_json
  (image_path -> width/height), instead of auto/default heuristics.
"""

import argparse
import json
import os
import time
from typing import Dict, List, Optional, Tuple

import torch
from accelerate import Accelerator
from PIL import Image


NEGATIVE_PROMPT = (
    "low quality, blurry, distorted, deformed, ugly, bad anatomy, "
    "bad hands, extra fingers, missing fingers, extra limbs, "
    "disfigured, watermark, text, logo, signature, cropped, "
    "oversaturated, underexposed, overexposed, noise, grain, "
    "cartoon, anime, illustration, 3d render, cgi"
)


def str2bool(v):
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid bool value: {v}")


def build_prompt(content_desc: str, composition_analysis: str) -> str:
    return f"{content_desc.strip()}\n\nComposition analysis: {composition_analysis.strip()}"


def get_prompt(item: Dict) -> str:
    if "prompt" in item and str(item["prompt"]).strip():
        return str(item["prompt"]).strip()
    return build_prompt(item.get("content_description", ""), item.get("composition_analysis", ""))


def get_portrait_resolution(content_desc: str) -> Tuple[int, int]:
    desc_lower = content_desc.lower()
    is_vertical = any(
        kw in desc_lower
        for kw in [
            "standing",
            "full body",
            "full-length",
            "head to toe",
            "tall",
            "vertical",
            "portrait orientation",
        ]
    )
    is_horizontal = any(
        kw in desc_lower
        for kw in [
            "landscape",
            "panoramic",
            "wide shot",
            "horizontal",
            "lying down",
            "reclining",
            "group photo",
        ]
    )
    is_closeup = any(
        kw in desc_lower
        for kw in [
            "close-up",
            "closeup",
            "headshot",
            "face",
            "bust",
            "head and shoulders",
        ]
    )

    if is_closeup:
        return (896, 1152)
    if is_vertical:
        return (832, 1216)
    if is_horizontal:
        return (1216, 832)
    return (1024, 1024)


def load_test_items(test_json_path: str) -> List[Dict]:
    json_files: List[str] = []
    if os.path.isdir(test_json_path):
        for root, _, files in os.walk(test_json_path):
            for name in files:
                if name.lower().endswith(".json"):
                    json_files.append(os.path.join(root, name))
        json_files.sort()
    else:
        json_files = [test_json_path]

    all_items: List[Dict] = []
    for jf in json_files:
        with open(jf, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            all_items.extend(data)
        elif isinstance(data, dict):
            if isinstance(data.get("data"), list):
                all_items.extend(data["data"])
            else:
                raise ValueError(f"Unsupported JSON structure in {jf}")
        else:
            raise ValueError(f"Unsupported JSON structure in {jf}")
    return all_items


def normalize_rel_path(image_path: str) -> str:
    return image_path.strip().lstrip("/\\")


def load_predicted_sizes(predicted_json_path: str) -> Dict[str, Tuple[int, int]]:
    with open(predicted_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("--predicted_sizes_json must be a JSON array")

    size_map: Dict[str, Tuple[int, int]] = {}
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        image_path = str(item.get("image_path", "")).strip()
        width = item.get("width")
        height = item.get("height")
        if not image_path:
            continue
        if width is None or height is None:
            continue
        try:
            w = int(width)
            h = int(height)
        except (TypeError, ValueError):
            raise ValueError(f"Invalid width/height at predicted item #{idx}") from None
        if w <= 0 or h <= 0:
            raise ValueError(f"Non-positive width/height at predicted item #{idx}: {w}x{h}")

        key_raw = image_path
        key_norm = normalize_rel_path(image_path)
        size_map[key_raw] = (w, h)
        size_map[key_norm] = (w, h)
    return size_map


def load_zimage_text2img_pipe(model_path: str, device: str, lora_path: Optional[str], lora_scale: float):
    from diffusers import ZImagePipeline

    pipe = ZImagePipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
    )
    if lora_path:
        pipe.load_lora_weights(lora_path)
        pipe.fuse_lora(lora_scale=lora_scale)
        pipe.unload_lora_weights()
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def load_zimage_img2img_pipe(model_path: str, device: str, lora_path: Optional[str], lora_scale: float):
    from diffusers import ZImageImg2ImgPipeline

    pipe = ZImageImg2ImgPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
    )
    if lora_path:
        pipe.load_lora_weights(lora_path)
        pipe.fuse_lora(lora_scale=lora_scale)
        pipe.unload_lora_weights()
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def generate_with_lora(
    pipe,
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    steps: int,
    guidance: float,
    seed: int,
    device: str,
    cfg_normalization: bool,
    max_sequence_length: int,
) -> Image.Image:
    generator = torch.Generator(device=device).manual_seed(seed)
    with torch.inference_mode():
        out = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            cfg_normalization=cfg_normalization,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator,
            max_sequence_length=max_sequence_length,
        )
    return out.images[0]


def run_img2img(
    pipe,
    init_image: Image.Image,
    prompt: str,
    negative_prompt: str,
    strength: float,
    steps: int,
    guidance: float,
    seed: int,
    device: str,
    width: Optional[int],
    height: Optional[int],
    max_sequence_length: int,
) -> Image.Image:
    generator = torch.Generator(device=device).manual_seed(seed)
    kwargs = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "num_inference_steps": steps,
        "guidance_scale": guidance,
        "generator": generator,
        "strength": strength,
        "max_sequence_length": max_sequence_length,
    }
    if width is not None:
        kwargs["width"] = width
    if height is not None:
        kwargs["height"] = height

    image_rgb = init_image.convert("RGB")
    try:
        with torch.inference_mode():
            out = pipe(image=image_rgb, **kwargs)
    except TypeError:
        with torch.inference_mode():
            out = pipe(init_image=image_rgb, **kwargs)
    return out.images[0]


def resolve_reference_image(image_path: str, reference_image_dir: str) -> Optional[str]:
    rel_path = normalize_rel_path(image_path)
    candidates = [os.path.join(reference_image_dir, rel_path)]

    base = os.path.basename(rel_path)
    stem, _ = os.path.splitext(base)
    candidates.append(os.path.join(reference_image_dir, base))
    candidates.extend(
        [
            os.path.join(reference_image_dir, stem + ".png"),
            os.path.join(reference_image_dir, stem + ".jpg"),
            os.path.join(reference_image_dir, stem + ".jpeg"),
            os.path.join(reference_image_dir, stem + ".webp"),
        ]
    )

    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def read_image_rgb_and_size(image_path: str) -> Tuple[Image.Image, Tuple[int, int]]:
    with Image.open(image_path) as img:
        img_rgb = img.convert("RGB")
        size = img_rgb.size
    return img_rgb, size


def resolve_stage1_size(
    item: Dict,
    image_path: str,
    rel_out: str,
    predicted_sizes: Dict[str, Tuple[int, int]],
    auto_resolution: bool,
    fallback_width: int,
    fallback_height: int,
) -> Tuple[int, int, bool]:
    if image_path in predicted_sizes:
        w, h = predicted_sizes[image_path]
        return w, h, False
    if rel_out in predicted_sizes:
        w, h = predicted_sizes[rel_out]
        return w, h, False

    if auto_resolution:
        w, h = get_portrait_resolution(item.get("content_description", ""))
    else:
        w, h = fallback_width, fallback_height
    return w, h, True


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Two-stage inference: ZImage+LoRA then ZImage img2img. "
            "Stage-1 resolution is read from --predicted_sizes_json."
        )
    )
    parser.add_argument("--test_json", type=str, required=True, help="A json file or a directory containing json files")
    parser.add_argument("--predicted_sizes_json", type=str, required=True, help="JSON list containing image_path/width/height")
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, default="/data/wangxinghao/models/Z-Image")
    parser.add_argument(
        "--img2img_model_path",
        type=str,
        default="",
        help="Model path for img2img stage (e.g. Z-Image Turbo). If empty, defaults to --model_path.",
    )
    parser.add_argument("--lora_path", type=str, default="")
    parser.add_argument(
        "--img2img_lora_path",
        type=str,
        default="",
        help="LoRA path for img2img stage. If empty, defaults to --lora_path.",
    )
    parser.add_argument("--lora_scale", type=float, default=1.0)
    parser.add_argument("--use_lora_gen", type=str2bool, default=True)
    parser.add_argument("--use_lora_img2img", type=str2bool, default=True)
    parser.add_argument("--reference_image_dir", type=str, default="")
    parser.add_argument(
        "--stage1_output_path",
        type=str,
        default="",
        help="Required when --reference_image_dir is empty. Stores stage-1 generated images.",
    )
    parser.add_argument("--steps", type=int, default=50, help="ZImage+LoRA stage steps")
    parser.add_argument("--img2img_steps", type=int, default=50, help="img2img stage steps")
    parser.add_argument("--guidance_scale", type=float, default=4.0)
    parser.add_argument("--img2img_guidance_scale", type=float, default=4.0)
    parser.add_argument("--strength", type=float, default=0.99, help="img2img strength")
    parser.add_argument("--max_sequence_length", type=int, default=768)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--auto_resolution", action="store_true")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--use_negative_prompt_stage1", type=str2bool, default=True)
    parser.add_argument("--use_negative_prompt_stage2", type=str2bool, default=False)
    parser.add_argument("--cfg_normalization", action="store_true")
    parser.add_argument("--batch_start", type=int, default=0)
    parser.add_argument("--batch_end", type=int, default=-1)
    parser.add_argument("--save_prompts", action="store_true")
    args = parser.parse_args()

    accelerator = Accelerator()
    device = accelerator.device
    device_str = str(device)
    rank = accelerator.process_index
    world = accelerator.num_processes

    t_all_start = time.time()
    if accelerator.is_main_process:
        os.makedirs(args.output_path, exist_ok=True)

    if accelerator.is_main_process:
        print(
            f"[INFO] Accelerate | world_size={world} | rank={rank} | device={device}",
            flush=True,
        )
        print(f"[INFO] Loading test data from: {args.test_json}", flush=True)
        print(f"[INFO] Loading predicted sizes from: {args.predicted_sizes_json}", flush=True)

    items = load_test_items(args.test_json)
    predicted_sizes = load_predicted_sizes(args.predicted_sizes_json)
    end = args.batch_end if args.batch_end > 0 else len(items)
    items = items[args.batch_start:end]
    n_items = len(items)

    my_indices = list(range(rank, n_items, world))
    if accelerator.is_main_process:
        print(
            f"[INFO] Total items in batch: {n_items} "
            f"(each rank handles ~{n_items // world + (1 if n_items % world > 0 else 0)})",
            flush=True,
        )

    if args.save_prompts and accelerator.is_main_process:
        prompts_path = os.path.join(args.output_path, "prompts.json")
        prompts_log = [{"image_path": item.get("image_path", ""), "prompt": get_prompt(item)} for item in items]
        with open(prompts_path, "w", encoding="utf-8") as f:
            json.dump(prompts_log, f, indent=2, ensure_ascii=False)
        print(f"[INFO] Prompts saved to: {prompts_path}", flush=True)

    accelerator.wait_for_everyone()

    has_reference_dir = bool(args.reference_image_dir and os.path.isdir(args.reference_image_dir))
    if args.reference_image_dir and not has_reference_dir and accelerator.is_main_process:
        print(
            f"[WARN] reference_image_dir does not exist: {args.reference_image_dir}. "
            "Will use generate->img2img mode for all items.",
            flush=True,
        )
    if not has_reference_dir:
        if not args.stage1_output_path:
            raise ValueError(
                "--stage1_output_path must be provided when --reference_image_dir is empty or invalid."
            )
        if accelerator.is_main_process:
            os.makedirs(args.stage1_output_path, exist_ok=True)

    img2img_model_path = args.img2img_model_path if args.img2img_model_path else args.model_path
    text2img_lora_path = args.lora_path if (args.use_lora_gen and args.lora_path) else None
    img2img_lora_path_raw = args.img2img_lora_path if args.img2img_lora_path else args.lora_path
    img2img_lora_path = img2img_lora_path_raw if (args.use_lora_img2img and img2img_lora_path_raw) else None

    pipe_lora = None
    if not has_reference_dir:
        if accelerator.is_main_process:
            print(
                f"[INFO] Loading stage-1 text2img pipeline... (use_lora_gen={args.use_lora_gen}, "
                f"LoRA: {text2img_lora_path if text2img_lora_path else 'None'})",
                flush=True,
            )
        pipe_lora = load_zimage_text2img_pipe(
            model_path=args.model_path,
            device=device_str,
            lora_path=text2img_lora_path,
            lora_scale=args.lora_scale,
        )
    if accelerator.is_main_process:
        print(
            "[INFO] Loading ZImage img2img pipeline... "
            f"(model: {img2img_model_path}, use_lora_img2img={args.use_lora_img2img}, "
            f"LoRA: {img2img_lora_path if img2img_lora_path else 'None'})",
            flush=True,
        )
    pipe_img2img = load_zimage_img2img_pipe(
        model_path=img2img_model_path,
        device=device_str,
        lora_path=img2img_lora_path,
        lora_scale=args.lora_scale,
    )

    accelerator.wait_for_everyone()
    negative_prompt_stage1 = NEGATIVE_PROMPT if args.use_negative_prompt_stage1 else ""
    negative_prompt_stage2 = NEGATIVE_PROMPT if args.use_negative_prompt_stage2 else ""

    direct_from_reference_count = 0
    generate_then_img2img_count = 0
    stage1_saved_count = 0
    stage1_skipped_existing_count = 0
    skipped_existing_count = 0
    resize_from_reference_count = 0
    missing_pred_size_count = 0

    if not has_reference_dir:
        for local_k, idx in enumerate(my_indices):
            item = items[idx]
            image_path = item.get("image_path", f"sample_{idx:06d}.png")
            rel_out = normalize_rel_path(image_path)
            stage1_file = os.path.join(args.stage1_output_path, rel_out)
            os.makedirs(os.path.dirname(stage1_file) or ".", exist_ok=True)

            if os.path.exists(stage1_file):
                stage1_skipped_existing_count += 1
                continue

            prompt = get_prompt(item)
            seed_i = args.seed + idx * 1000
            stage1_w, stage1_h, missing_flag = resolve_stage1_size(
                item=item,
                image_path=image_path,
                rel_out=rel_out,
                predicted_sizes=predicted_sizes,
                auto_resolution=args.auto_resolution,
                fallback_width=args.width,
                fallback_height=args.height,
            )
            if missing_flag:
                missing_pred_size_count += 1
                print(
                    f"[WARN] rank={rank} missing predicted size for image_path='{image_path}', "
                    f"fallback to {stage1_w}x{stage1_h}",
                    flush=True,
                )

            print(
                f"[INFO] rank={rank} [stage1 {local_k + 1}/{len(my_indices)}] global #{idx + 1}/{n_items} "
                f"generate -> '{stage1_file}', size={stage1_w}x{stage1_h}",
                flush=True,
            )
            stage1_img = generate_with_lora(
                pipe=pipe_lora,
                prompt=prompt,
                negative_prompt=negative_prompt_stage1,
                width=stage1_w,
                height=stage1_h,
                steps=args.steps,
                guidance=args.guidance_scale,
                seed=seed_i,
                device=device_str,
                cfg_normalization=args.cfg_normalization,
                max_sequence_length=args.max_sequence_length,
            )
            if stage1_file.lower().endswith(".png"):
                stage1_img.save(stage1_file, "PNG")
            else:
                stage1_img.save(stage1_file, "JPEG", quality=95)
            stage1_saved_count += 1

        accelerator.wait_for_everyone()

    for local_k, idx in enumerate(my_indices):
        item = items[idx]
        image_path = item.get("image_path", f"sample_{idx:06d}.png")
        rel_out = normalize_rel_path(image_path)
        output_file = os.path.join(args.output_path, rel_out)
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

        if os.path.exists(output_file):
            skipped_existing_count += 1
            continue

        prompt = get_prompt(item)
        seed_i = args.seed + idx * 1000

        default_w, default_h, missing_flag = resolve_stage1_size(
            item=item,
            image_path=image_path,
            rel_out=rel_out,
            predicted_sizes=predicted_sizes,
            auto_resolution=args.auto_resolution,
            fallback_width=args.width,
            fallback_height=args.height,
        )
        if missing_flag and has_reference_dir:
            missing_pred_size_count += 1
            print(
                f"[WARN] rank={rank} missing predicted size for image_path='{image_path}', "
                f"fallback to {default_w}x{default_h}",
                flush=True,
            )

        target_w, target_h = default_w, default_h

        init_img: Optional[Image.Image] = None
        if has_reference_dir:
            ref_img_path = resolve_reference_image(rel_out, args.reference_image_dir)
            if ref_img_path:
                init_img, ref_size = read_image_rgb_and_size(ref_img_path)
                target_w, target_h = ref_size
                resize_from_reference_count += 1
                print(
                    f"[INFO] rank={rank} [{local_k + 1}/{len(my_indices)}] global #{idx + 1}/{n_items} "
                    f"source: '{ref_img_path}' (reference folder -> img2img), target_size={target_w}x{target_h}",
                    flush=True,
                )
                direct_from_reference_count += 1

        if init_img is None:
            if has_reference_dir:
                print(
                    f"[INFO] rank={rank} [{local_k + 1}/{len(my_indices)}] global #{idx + 1}/{n_items} "
                    f"source: ZImage+LoRA then img2img, stage1={default_w}x{default_h}, "
                    f"target_size={target_w}x{target_h}",
                    flush=True,
                )
                init_img = generate_with_lora(
                    pipe=pipe_lora,
                    prompt=prompt,
                    negative_prompt=negative_prompt_stage1,
                    width=default_w,
                    height=default_h,
                    steps=args.steps,
                    guidance=args.guidance_scale,
                    seed=seed_i,
                    device=device_str,
                    cfg_normalization=args.cfg_normalization,
                    max_sequence_length=args.max_sequence_length,
                )
            else:
                stage1_file = os.path.join(args.stage1_output_path, rel_out)
                if not os.path.exists(stage1_file):
                    raise FileNotFoundError(
                        f"Stage-1 image missing for rank={rank}, idx={idx}, expected: {stage1_file}"
                    )
                init_img, _ = read_image_rgb_and_size(stage1_file)
                print(
                    f"[INFO] rank={rank} [{local_k + 1}/{len(my_indices)}] global #{idx + 1}/{n_items} "
                    f"source: '{stage1_file}' (stage1 folder -> img2img), target_size={target_w}x{target_h}",
                    flush=True,
                )
            generate_then_img2img_count += 1

        init_img = init_img.resize((target_w, target_h), Image.Resampling.LANCZOS)
        out_img = run_img2img(
            pipe=pipe_img2img,
            init_image=init_img,
            prompt=prompt,
            negative_prompt=negative_prompt_stage2,
            strength=args.strength,
            steps=args.img2img_steps,
            guidance=args.img2img_guidance_scale,
            seed=seed_i + 17,
            device=device_str,
            width=target_w,
            height=target_h,
            max_sequence_length=args.max_sequence_length,
        )

        if output_file.lower().endswith(".png"):
            out_img.save(output_file, "PNG")
        else:
            out_img.save(output_file, "JPEG", quality=95)

    accelerator.wait_for_everyone()
    total_time_sec = time.time() - t_all_start

    counts = torch.tensor(
        [
            float(direct_from_reference_count),
            float(generate_then_img2img_count),
            float(stage1_saved_count),
            float(stage1_skipped_existing_count),
            float(skipped_existing_count),
            float(resize_from_reference_count),
            float(missing_pred_size_count),
        ],
        dtype=torch.float64,
        device=device,
    )
    counts_sum = accelerator.reduce(counts, reduction="sum")
    if accelerator.is_main_process:
        d_direct = int(counts_sum[0].item())
        d_gen = int(counts_sum[1].item())
        d_stage1_saved = int(counts_sum[2].item())
        d_stage1_skip = int(counts_sum[3].item())
        d_skip = int(counts_sum[4].item())
        d_resize = int(counts_sum[5].item())
        d_missing = int(counts_sum[6].item())
        processed_count = d_direct + d_gen
        print("\n========== Inference Summary ==========", flush=True)
        print(
            f"Total elapsed time (wall, main process): {total_time_sec:.2f} s "
            f"({total_time_sec / 60.0:.2f} min)",
            flush=True,
        )
        print(f"World size: {world}", flush=True)
        print(f"Processed samples: {processed_count}", flush=True)
        print(f"Directly from reference image folder: {d_direct}", flush=True)
        print(f"Generate (ZImage+LoRA) then img2img: {d_gen}", flush=True)
        print(f"Stage-1 saved images: {d_stage1_saved}", flush=True)
        print(f"Stage-1 skipped existing images: {d_stage1_skip}", flush=True)
        print(f"Used reference size for img2img target: {d_resize}", flush=True)
        print(f"Missing predicted-size fallback count: {d_missing}", flush=True)
        print(f"Skipped existing outputs: {d_skip}", flush=True)
        if not has_reference_dir:
            print(f"Stage-1 outputs saved to: {args.stage1_output_path}", flush=True)
        print(f"Outputs saved to: {args.output_path}", flush=True)


if __name__ == "__main__":
    main()
