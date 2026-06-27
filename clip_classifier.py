#!/usr/bin/env python3
"""
CLIP-based classifier to split "玩游戏" images into:
- 玩手机 (playing on phone)
- 玩电脑 (playing on computer)

Workflow:
1) Scores each image with CLIP using multilingual prompts (CN + EN).
2) Writes a CSV with per-image probabilities & confidence.
3) Copies images into output folders based on confidence:
   - phone/high_conf
   - computer/high_conf
   - review/medium_conf
   - review/low_conf

You can then review only the "review/*" folders (especially low_conf).

Usage:
  python clip_classifier.py --input /path/to/images --output /path/to/output \
      --high 0.80 --low 0.60 --model ViT-B/32 --device auto

Install deps (one-time):
  pip install torch torchvision ftfy regex tqdm pandas pillow
  # Install CLIP (official repo)
  pip install git+https://github.com/openai/CLIP.git

Notes:
- "high" and "low" thresholds can be tuned. Start with --high 0.80 --low 0.60.
- Confidence = max(prob_phone, prob_computer).
- Multi-lingual prompt ensembling helps robustness.
"""

import argparse
import csv
import os
import shutil
from pathlib import Path
from typing import List, Tuple

import torch
import clip  # from openai/CLIP
from PIL import Image
import pandas as pd
from tqdm import tqdm


# ----- Prompt sets (Chinese + English, with small variations) -----
PHONE_PROMPTS = [
    "一个人在用手机打游戏",
    "一个人在玩手游",
    "一个人低头看着手机屏幕玩游戏",
    "手机游戏场景",
    "a person playing a game on a phone",
    "mobile gaming, person using a smartphone",
]
COMPUTER_PROMPTS = [
    "一个人在用电脑打游戏",
    "一个人在玩电脑游戏",
    "一个人坐在电脑前玩游戏，有键盘或鼠标",
    "电脑游戏场景",
    "a person playing a game on a computer",
    "PC gaming, person at a computer with keyboard or mouse",
]


def load_model(model_name: str, device: str) -> Tuple[torch.nn.Module, torch.device, any]:
    if device == "auto":
        dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        dev = device
    device_t = torch.device(dev)
    model, preprocess = clip.load(model_name, device=device_t, jit=False)
    model.eval()
    return model, device_t, preprocess


def encode_texts(model, device, prompts: List[str]) -> torch.Tensor:
    with torch.no_grad():
        tokens = clip.tokenize(prompts).to(device)
        text_features = model.encode_text(tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features


def score_image(model, device, preprocess, image_path: Path, phone_text: torch.Tensor, comp_text: torch.Tensor) -> Tuple[float, float]:
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception:
        return float("nan"), float("nan")
    image_input = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # cosine similarity to the mean text embedding for each class
        # (We average the embeddings from our prompt set for robustness)
        phone_sim = (image_features @ phone_text.T).squeeze(0)
        comp_sim  = (image_features @ comp_text.T).squeeze(0)

        # Average over prompts (logits). Then softmax across the two classes.
        phone_logit = phone_sim.mean().unsqueeze(0)  # shape [1]
        comp_logit  = comp_sim.mean().unsqueeze(0)   # shape [1]
        logits = torch.stack([phone_logit, comp_logit], dim=-1)  # [1, 2]
        probs = logits.softmax(dim=-1).squeeze(0).tolist()       # [2]
        prob_phone, prob_comp = float(probs[0]), float(probs[1])
    return prob_phone, prob_comp


def should_copy_to_folder(prob_phone: float, prob_comp: float, high_thr: float, low_thr: float) -> Tuple[str, str]:
    """Return (split, confidence_bucket). split in {'phone','computer','review'}; confidence_bucket in {'high_conf','medium_conf','low_conf'}"""
    if any(map(lambda x: x != x, [prob_phone, prob_comp])):  # NaN check
        return "review", "low_conf"
    conf = max(prob_phone, prob_comp)
    if conf >= high_thr:
        return ("phone" if prob_phone >= prob_comp else "computer"), "high_conf"
    elif conf >= low_thr:
        return "review", "medium_conf"
    else:
        return "review", "low_conf"


def is_image_file(p: Path) -> bool:
    return p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input folder with images (scans recursively).")
    ap.add_argument("--output", required=True, help="Output folder.")
    ap.add_argument("--model", default="ViT-B/32", help="CLIP model name (e.g., ViT-B/32, ViT-L/14).")
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    ap.add_argument("--high", type=float, default=0.80, help="High-confidence threshold.")
    ap.add_argument("--low", type=float, default=0.60, help="Low-confidence threshold.")
    ap.add_argument("--copy", action="store_true", help="Copy files (default is to MOVE).")
    ap.add_argument("--max", type=int, default=0, help="Max number of images to process (0 = all).")
    args = ap.parse_args()

    in_dir = Path(args.input).expanduser()
    out_dir = Path(args.output).expanduser()

    out_phone_high = out_dir / "phone" / "high_conf"
    out_comp_high = out_dir / "computer" / "high_conf"
    out_rev_med = out_dir / "review" / "medium_conf"
    out_rev_low = out_dir / "review" / "low_conf"
    for d in [out_phone_high, out_comp_high, out_rev_med, out_rev_low]:
        d.mkdir(parents=True, exist_ok=True)

    model, device, preprocess = load_model(args.model, args.device)

    # Prepare text embeddings (mean of prompts for each class)
    phone_text_feats = encode_texts(model, device, PHONE_PROMPTS)
    comp_text_feats  = encode_texts(model, device, COMPUTER_PROMPTS)

    # Gather images
    images = [p for p in in_dir.rglob("*") if p.is_file() and is_image_file(p)]
    if args.max > 0:
        images = images[:args.max]

    records = []
    for p in tqdm(images, desc="Scoring images"):
        prob_phone, prob_comp = score_image(model, device, preprocess, p, phone_text_feats, comp_text_feats)
        split, bucket = should_copy_to_folder(prob_phone, prob_comp, args.high, args.low)

        # Decide destination
        if split == "phone" and bucket == "high_conf":
            dest_dir = out_phone_high
            predicted = "phone"
        elif split == "computer" and bucket == "high_conf":
            dest_dir = out_comp_high
            predicted = "computer"
        else:
            # review buckets
            dest_dir = out_rev_med if bucket == "medium_conf" else out_rev_low
            predicted = "review"

        dest_path = dest_dir / p.name
        try:
            if args.copy:
                shutil.copy2(p, dest_path)
            else:
                shutil.move(str(p), dest_path)
        except Exception as e:
            # If moving fails (e.g., cross-device), fallback to copy
            try:
                shutil.copy2(p, dest_path)
            except Exception:
                pass  # ignore file I/O errors in batch

        conf = max(prob_phone if prob_phone == prob_phone else 0.0,
                   prob_comp if prob_comp == prob_comp else 0.0)

        records.append({
            "file": str(dest_path),
            "predicted": predicted,
            "prob_phone": prob_phone,
            "prob_computer": prob_comp,
            "confidence": conf,
            "bucket": bucket
        })

    # Save CSV summary
    df = pd.DataFrame(records)
    csv_path = out_dir / "clip_scores.csv"
    df.to_csv(csv_path, index=False, quoting=csv.QUOTE_NONNUMERIC)
    print(f"Saved results to: {csv_path}")
    print("Folder layout:")
    print(f"  {out_phone_high}")
    print(f"  {out_comp_high}")
    print(f"  {out_rev_med}")
    print(f"  {out_rev_low}")
    print("Tip: Review low_conf first, then medium_conf.")

if __name__ == "__main__":
    main()
