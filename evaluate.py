"""
Test-set inference with Qwen3-VL + CER / WER evaluation.

Usage
-----
Edit the CONFIG block at the bottom of this file, then run:

    python evaluate.py

Outputs
-------
- A JSON file with per-image predictions, CER, WER, and a summary row.
- A plain-text log file (mirrors everything printed to the terminal).

Results on NAKBA 2026 test set (Qwen3-VL-4B, 32 epochs):
    Avg CER : 8.59 %
    Avg WER : 25.87 %
"""

import os
import sys
import json
import time
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

import torch
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

from prompts import third_prompt


# ── Logger ────────────────────────────────────────────────────────────────────

class TeeLogger:
    """Write to both the terminal and a log file simultaneously."""

    def __init__(self, log_file: str):
        self.terminal = sys.stdout
        self.log = open(log_file, "a", encoding="utf-8")

    def write(self, msg):
        self.terminal.write(msg)
        self.log.write(msg)
        self.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


# ── OCR engine ────────────────────────────────────────────────────────────────

class Qwen3VLOCR:
    """Thin wrapper around Qwen3-VL for single-image OCR."""

    def __init__(self, model_path: str, torch_dtype=torch.bfloat16, verbose: bool = True):
        self.verbose = verbose
        if verbose:
            print(f"Loading model: {model_path}")

        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, dtype=torch_dtype, device_map="auto"
        ).eval()

        if verbose:
            print("Model loaded.")

    def extract_text(
        self,
        image_path: str,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
        enable_retry: bool = False,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> Dict[str, Any]:
        """Return a dict with 'text', 'retries', and 'status'."""

        for attempt in range(max_retries + 1 if enable_retry else 1):
            try:
                image = Image.open(image_path).convert("RGB")
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]
                text_input = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = self.processor(
                    text=[text_input], images=[image], padding=True, return_tensors="pt"
                ).to(self.model.device)

                gen_kwargs = {"max_new_tokens": max_tokens, "do_sample": temperature > 0}
                if temperature > 0:
                    gen_kwargs["temperature"] = temperature

                with torch.no_grad():
                    generated_ids = self.model.generate(**inputs, **gen_kwargs)

                output_ids = generated_ids[:, inputs["input_ids"].shape[1]:]
                text = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]

                return {"text": text.strip(), "retries": attempt, "status": "success"}

            except Exception as exc:
                if enable_retry and attempt < max_retries:
                    if self.verbose:
                        print(f"  Retry {attempt + 1}/{max_retries}: {exc}")
                    time.sleep(retry_delay)
                else:
                    raise


# ── Metrics ───────────────────────────────────────────────────────────────────

def _levenshtein(a, b):
    """Generic Levenshtein distance over sequences."""
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[j] = prev[j - 1]
            else:
                dp[j] = 1 + min(prev[j], dp[j - 1], prev[j - 1])
    return dp[n]


def calculate_cer(reference: str, hypothesis: str) -> Optional[float]:
    if not reference:
        return 0.0 if not hypothesis else 100.0
    if not hypothesis:
        return 100.0
    return round(min(_levenshtein(reference, hypothesis) / len(reference), 1.0) * 100, 2)


def calculate_wer(reference: str, hypothesis: str) -> Optional[float]:
    ref_w, hyp_w = reference.split(), hypothesis.split()
    if not ref_w:
        return 0.0 if not hyp_w else 100.0
    if not hyp_w:
        return 100.0
    return round(min(_levenshtein(ref_w, hyp_w) / len(ref_w), 1.0) * 100, 2)


# ── Ground truth loader ───────────────────────────────────────────────────────

def load_ground_truth(csv_path: str) -> Dict[str, str]:
    """Load a CSV with columns 'image' and 'text' into a dict."""
    import csv

    if not csv_path or not Path(csv_path).exists():
        return {}
    gt = {}
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = row.get("image", "").strip()
            if name:
                gt[name] = row.get("text", "").strip()
    return gt


# ── Main processing ───────────────────────────────────────────────────────────

def evaluate(
    images_folder: str,
    output_json: str,
    ground_truth_csv: str = "",
    log_file: str = "",
    prompt: str = third_prompt,
    model_path: str = "",
    temperature: float = 0.0,
    max_tokens: int = 512,
    timeout_seconds: int = 60,
    enable_retry: bool = False,
    max_retries: int = 3,
    retry_delay: float = 1.0,
):
    # ── Setup paths ───────────────────────────────────────────────────────────
    out_path = Path(output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not log_file:
        log_file = str(out_path.parent / f"{out_path.stem}_log.txt")
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"{'='*80}\nNAKBA OCR Evaluation Log\nModel: {model_path}\n"
                f"Started: {datetime.now():%Y-%m-%d %H:%M:%S}\n{'='*80}\n\n")

    original_stdout = sys.stdout
    tee = TeeLogger(log_file)
    sys.stdout = tee

    try:
        start_time = time.time()

        gt = load_ground_truth(ground_truth_csv)
        print(f"Ground truth entries: {len(gt)}")

        img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        image_files = sorted(f for f in Path(images_folder).iterdir()
                             if f.is_file() and f.suffix.lower() in img_exts)
        if not image_files:
            print(f"No images found in {images_folder}")
            return

        total = len(image_files)
        print(f"\nImages to process : {total}")
        print(f"Model             : {model_path}")
        print(f"Ground truth      : {'yes' if gt else 'no'}")
        print(f"Output JSON       : {output_json}")
        print("=" * 80 + "\n")

        ocr = Qwen3VLOCR(model_path=model_path, verbose=True)

        results = []
        for idx, img_file in enumerate(image_files, 1):
            t0 = time.time()
            name = img_file.name
            try:
                res = ocr.extract_text(
                    image_path=str(img_file),
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    enable_retry=enable_retry,
                    max_retries=max_retries,
                    retry_delay=retry_delay,
                )
                elapsed = time.time() - t0
                pred = res["text"]
                ref = gt.get(name, "")
                cer = calculate_cer(ref, pred) if ref else None
                wer = calculate_wer(ref, pred) if ref else None

                results.append({
                    "image_name": name,
                    "original_transcript": ref,
                    "predicted_transcript": pred,
                    "CER": cer,
                    "WER": wer,
                    "processing_time": round(elapsed, 2),
                    "status": "success",
                    "retries": res.get("retries", 0),
                })
                metrics = f"CER:{cer:5.1f}%  WER:{wer:5.1f}%" if cer is not None else "no ground truth"
                retry_tag = f"  retries:{res['retries']}" if res.get("retries") else ""
                print(f"[{idx:4d}/{total}] ✓  {name:<40} | {elapsed:6.2f}s | {metrics}{retry_tag}")

            except Exception as exc:
                elapsed = time.time() - t0
                ref = gt.get(name, "")
                results.append({
                    "image_name": name,
                    "original_transcript": ref,
                    "predicted_transcript": "",
                    "CER": None, "WER": None,
                    "processing_time": round(elapsed, 2),
                    "status": "error",
                    "error": str(exc),
                })
                print(f"[{idx:4d}/{total}] ✗  {name}  ERROR: {exc}")

        # ── Aggregate metrics ─────────────────────────────────────────────────
        ok = [r for r in results if r["status"] == "success"]
        with_metrics = [r for r in ok if r["CER"] is not None]
        avg_cer = round(sum(r["CER"] for r in with_metrics) / len(with_metrics), 2) if with_metrics else None
        avg_wer = round(sum(r["WER"] for r in with_metrics) / len(with_metrics), 2) if with_metrics else None
        total_proc = round(sum(r["processing_time"] for r in ok), 2)

        summary = {"Avg_CER": avg_cer, "Avg_WER": avg_wer, "Total_processing_time": total_proc}
        output_data = results + [summary]

        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        elapsed_total = time.time() - start_time
        h, rem = divmod(elapsed_total, 3600)
        m, s = divmod(rem, 60)

        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"Total images  : {total}")
        print(f"Successful    : {len(ok)} ({len(ok)/total*100:.1f}%)")
        print(f"Failed        : {total - len(ok)}")
        if avg_cer is not None:
            print(f"Avg CER       : {avg_cer:.2f}%")
            print(f"Avg WER       : {avg_wer:.2f}%")
        print(f"Elapsed       : {int(h)}h {int(m)}m {s:.1f}s")
        print(f"Results saved : {output_json}")
        print("=" * 80)

    finally:
        sys.stdout = original_stdout
        tee.close()
        print(f"Log saved to: {log_file}")


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION – edit this block before running
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    evaluate(
        # ── Paths ──────────────────────────────────────────────────────────
        images_folder="./data/test/images",
        output_json="./output/test_results.json",
        ground_truth_csv="./data/test/annotations.csv",  # set "" to skip metrics
        log_file="./output/logs/test_eval.log",

        # ── Model ──────────────────────────────────────────────────────────
        model_path="./output/qwen3-vl-4b-arabic-ocr/final_model",

        # ── Inference settings ─────────────────────────────────────────────
        prompt=third_prompt,
        temperature=0.0,
        max_tokens=512,

        # ── Retry / timeout ────────────────────────────────────────────────
        timeout_seconds=60,
        enable_retry=True,
        max_retries=1,
        retry_delay=1.0,
    )
