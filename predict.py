"""
Blind test inference with Qwen3-VL – produces a submission CSV.

Usage
-----
Edit the CONFIG block at the bottom of this file, then run:

    python predict.py

Output
------
A CSV file with two columns:
    image   – filename of the input image
    text    – model prediction

Results on NAKBA 2026 blind set (Qwen3-VL-4B, 32 epochs):
    CER : 0.11 %
    WER : 0.31 %
"""

import sys
import csv
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


# ── Main processing ───────────────────────────────────────────────────────────

def predict(
    images_folder: str,
    output_csv: str,
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
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not log_file:
        log_file = str(out_path.parent / f"{out_path.stem}_log.txt")
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"{'='*80}\nNAKBA Blind Test Inference Log\nModel: {model_path}\n"
                f"Started: {datetime.now():%Y-%m-%d %H:%M:%S}\n{'='*80}\n\n")

    original_stdout = sys.stdout
    tee = TeeLogger(log_file)
    sys.stdout = tee

    try:
        start_time = time.time()

        img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        image_files = sorted(f for f in Path(images_folder).iterdir()
                             if f.is_file() and f.suffix.lower() in img_exts)
        if not image_files:
            print(f"No images found in {images_folder}")
            return

        total = len(image_files)
        print(f"Images to process : {total}")
        print(f"Model             : {model_path}")
        print(f"Output CSV        : {output_csv}")
        print("=" * 80 + "\n")

        ocr = Qwen3VLOCR(model_path=model_path, verbose=True)

        success_count = fail_count = 0

        with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["image", "text"])

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
                    writer.writerow([name, pred])
                    csvfile.flush()
                    success_count += 1
                    retry_tag = f"  retries:{res['retries']}" if res.get("retries") else ""
                    print(f"[{idx:4d}/{total}] ✓  {name:<40} | {elapsed:6.2f}s{retry_tag}")

                except Exception as exc:
                    elapsed = time.time() - t0
                    writer.writerow([name, ""])  # empty string for failed images
                    csvfile.flush()
                    fail_count += 1
                    print(f"[{idx:4d}/{total}] ✗  {name}  ERROR: {exc}")

        elapsed_total = time.time() - start_time
        h, rem = divmod(elapsed_total, 3600)
        m, s = divmod(rem, 60)

        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"Total images  : {total}")
        print(f"Successful    : {success_count} ({success_count/total*100:.1f}%)")
        print(f"Failed        : {fail_count}")
        print(f"Elapsed       : {int(h)}h {int(m)}m {s:.1f}s")
        print(f"Submission    : {output_csv}")
        print("=" * 80)

    finally:
        sys.stdout = original_stdout
        tee.close()
        print(f"Log saved to: {log_file}")


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION – edit this block before running
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    predict(
        # ── Paths ──────────────────────────────────────────────────────────
        images_folder="./data/blind/images",
        output_csv="./output/blind_submission.csv",
        log_file="./output/logs/blind_predict.log",

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
