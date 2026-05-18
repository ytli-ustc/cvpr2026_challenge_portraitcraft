import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
import torch.nn.functional as F
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f]


def load_scheme(data_dir):
    p = Path(data_dir) / "buckets.json"
    with open(p) as f:
        meta = json.load(f)
    letters = [c["letter"] for c in meta["classes"]]
    reps = [(c["h"], c["w"]) for c in meta["classes"]]
    return letters, reps


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="/local/liyanting/checkpoints/Qwen3-0.6B")
    ap.add_argument(
        "--lora_path",
        default="/local/liyanting/portrait/last_version/size_predictor_ckpt_leak/final",
    )
    ap.add_argument("--input_jsonl", required=True)
    ap.add_argument(
        "--data_dir",
        default=None,
        help="directory containing buckets.json. defaults to directory of --input_jsonl.",
    )
    ap.add_argument("--output_json", required=True)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=768)
    ap.add_argument("--no_lora", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device)
    model.eval()

    if not args.no_lora:
        model = PeftModel.from_pretrained(model, args.lora_path)
        model = model.merge_and_unload()
        model.eval()

    data_dir = args.data_dir or str(Path(args.input_jsonl).parent)
    letters, reps = load_scheme(data_dir)

    letter_ids = []
    for letter in letters:
        ids = tok.encode(" " + letter, add_special_tokens=False)
        assert len(ids) == 1, f"letter {letter!r} -> {ids}"
        letter_ids.append(ids[0])
    letter_ids_t = torch.tensor(letter_ids, device=device)

    items = load_jsonl(args.input_jsonl)
    output_items = []

    for start in tqdm(range(0, len(items), args.batch_size), desc="predict"):
        batch = items[start:start + args.batch_size]
        chats = [
            tok.apply_chat_template(
                [{"role": "user", "content": it["prompt"]}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for it in batch
        ]
        enc = tok(
            chats,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_len,
            add_special_tokens=False,
        ).to(device)

        out = model(**enc, use_cache=False)
        logits_last = out.logits[:, -1, :]
        letter_logits = logits_last[:, letter_ids_t]
        probs = F.softmax(letter_logits.float(), dim=-1)
        preds = letter_logits.argmax(dim=-1).tolist()

        for it, pred_idx, pred_probs in zip(batch, preds, probs.tolist()):
            pred_h, pred_w = reps[pred_idx]
            probs_rounded = [round(x, 4) for x in pred_probs]
            output_items.append(
                {
                    "image_path": it["image_path"],
                    "content_description": it.get("content_description", ""),
                    "composition_analysis": it.get("composition_analysis", ""),
                    "width": pred_w,
                    "height": pred_h,
                    "predicted_letter": letters[pred_idx],
                    "predicted_bucket": pred_idx,
                    "predicted_probs": probs_rounded,
                }
            )

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output_items, f, indent=2, ensure_ascii=False)
    print(f"wrote JSON: {out_path} ({len(output_items)} entries)")


if __name__ == "__main__":
    main()
