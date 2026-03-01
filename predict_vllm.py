"""
Blind test inference with Qwen3-VL via vLLM (parallel, faster than HuggingFace).

Usage
-----
1. Start the vLLM server (see README – "Serving the model").
2. Edit the CONFIG block at the bottom of this file.
3. Run:

    python predict_vllm.py

Output
------
A CSV file with two columns:
    image   – filename of the input image
    text    – model prediction

This script parallelises requests across `num_processes` workers, making it
significantly faster than the single-GPU HuggingFace approach when the vLLM
server has enough throughput.

Results on NAKBA 2026 blind set (Qwen3-VL-4B, 32 epochs, vLLM):
    CER : 0.11 %
    WER : 0.31 %
"""

import os
import sys
import csv
import base64
import time
import traceback
from pathlib import Path
from datetime import datetime
from multiprocessing import Pool
from typing import Dict, Any, Optional

import requests
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


# ── OCR client ────────────────────────────────────────────────────────────────

class Qwen3VLvLLMOCR:
    """
    HTTP client for a Qwen3-VL model served via vLLM's OpenAI-compatible API.

    The server must be running before calling this class.  See README for the
    exact `vllm serve` command.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        model_name: str = "qwen3-vl-4b",
        verbose: bool = True,
    ):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.verbose = verbose
        self.endpoint = f"{self.base_url}/v1/chat/completions"

    def _encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def extract_text(
        self,
        image_path: str,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
        timeout: Optional[float] = None,
        enable_retry: bool = False,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> Dict[str, Any]:
        """Return a dict with 'text', 'retries', and 'status'."""

        image_b64 = self._encode_image(image_path)
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        for attempt in range(max_retries + 1 if enable_retry else 1):
            try:
                resp = requests.post(
                    self.endpoint,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=timeout,
                )
                if not resp.ok:
                    try:
                        detail = resp.json()
                    except Exception:
                        detail = resp.text
                    raise Exception(f"API Error {resp.status_code}: {detail}")

                text = resp.json()["choices"][0]["message"]["content"]
                return {"text": text.strip(), "retries": attempt, "status": "success"}

            except Exception as exc:
                if enable_retry and attempt < max_retries:
                    if self.verbose:
                        print(f"  Retry {attempt + 1}/{max_retries}: {exc}")
                    time.sleep(retry_delay)
                else:
                    raise


# ── Worker function (runs in separate process) ────────────────────────────────

def _worker(args: tuple) -> dict:
    """Process a single image; designed to run in a multiprocessing Pool."""
    (
        image_file, prompt, base_url, model_name,
        temperature, max_tokens, timeout_seconds,
        enable_retry, max_retries, retry_delay,
    ) = args

    t0 = time.time()
    # Suppress stdout inside the worker
    old_stdout, sys.stdout = sys.stdout, open(os.devnull, "w")
    try:
        ocr = Qwen3VLvLLMOCR(base_url=base_url, model_name=model_name, verbose=False)
        sys.stdout.close()
        sys.stdout = old_stdout

        res = ocr.extract_text(
            image_path=str(image_file),
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout_seconds or None,
            enable_retry=enable_retry,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )
        return {
            "image_name": image_file.name,
            "text": res["text"],
            "retries": res.get("retries", 0),
            "elapsed": round(time.time() - t0, 2),
            "status": "success",
        }

    except Exception as exc:
        if sys.stdout != old_stdout:
            sys.stdout.close()
            sys.stdout = old_stdout
        return {
            "image_name": image_file.name,
            "text": "",
            "elapsed": round(time.time() - t0, 2),
            "status": "error",
            "error": str(exc),
        }


# ── Main processing ───────────────────────────────────────────────────────────

def predict_vllm(
    images_folder: str,
    output_csv: str,
    log_file: str = "",
    prompt: str = third_prompt,
    base_url: str = "http://localhost:8000",
    model_name: str = "qwen3-vl-4b",
    temperature: float = 0.0,
    max_tokens: int = 512,
    timeout_seconds: int = 60,
    enable_retry: bool = True,
    max_retries: int = 1,
    retry_delay: float = 1.0,
    num_processes: int = 10,
):
    # ── Setup paths ───────────────────────────────────────────────────────────
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not log_file:
        log_file = str(out_path.parent / f"{out_path.stem}_log.txt")
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    with open(log_file, "w", encoding="utf-8") as f:
        f.write(
            f"{'='*80}\nNAKBA Blind Test Inference Log (vLLM)\n"
            f"Model: {model_name}  Endpoint: {base_url}\n"
            f"Started: {datetime.now():%Y-%m-%d %H:%M:%S}\n{'='*80}\n\n"
        )

    original_stdout = sys.stdout
    tee = TeeLogger(log_file)
    sys.stdout = tee

    try:
        start_time = time.time()

        img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        image_files = sorted(
            f for f in Path(images_folder).iterdir()
            if f.is_file() and f.suffix.lower() in img_exts
        )
        if not image_files:
            print(f"No images found in {images_folder}")
            return

        total = len(image_files)
        print(f"Images to process : {total}")
        print(f"Model             : {model_name}")
        print(f"Endpoint          : {base_url}")
        print(f"Workers           : {num_processes}")
        print(f"Output CSV        : {output_csv}")
        print("=" * 80 + "\n")

        args_list = [
            (img, prompt, base_url, model_name, temperature, max_tokens,
             timeout_seconds, enable_retry, max_retries, retry_delay)
            for img in image_files
        ]

        success_count = fail_count = 0

        with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["image", "text"])

            print(f"Starting parallel processing ({num_processes} workers)...\n")

            with Pool(processes=num_processes) as pool:
                async_results = [pool.apply_async(_worker, (a,)) for a in args_list]

                for idx, (img_file, async_res) in enumerate(zip(image_files, async_results), 1):
                    name = img_file.name
                    try:
                        r = async_res.get()
                        writer.writerow([name, r["text"]])
                        csvfile.flush()

                        if r["status"] == "success":
                            success_count += 1
                            retry_tag = f"  retries:{r['retries']}" if r.get("retries") else ""
                            print(f"[{idx:4d}/{total}] ✓  {name:<40} | {r['elapsed']:6.2f}s{retry_tag}")
                        else:
                            fail_count += 1
                            print(f"[{idx:4d}/{total}] ✗  {name}  ERROR: {r.get('error', '?')}")

                    except Exception as exc:
                        writer.writerow([name, ""])
                        csvfile.flush()
                        fail_count += 1
                        print(f"[{idx:4d}/{total}] ✗  {name}  EXCEPTION: {exc}")

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

    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        raise

    finally:
        sys.stdout = original_stdout
        tee.close()
        print(f"Log saved to: {log_file}")


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION – edit this block before running
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    # ── Test mode ──────────────────────────────────────────────────────────
    # Set to True to verify the vLLM server is reachable before a full run.
    TEST_MODE = False

    # ── Paths ──────────────────────────────────────────────────────────────
    IMAGES_FOLDER = "./data/blind/images"
    OUTPUT_CSV    = "./output/blind_submission_vllm.csv"
    LOG_FILE      = "./output/logs/blind_predict_vllm.log"

    # ── vLLM server ────────────────────────────────────────────────────────
    # Must match the --served-model-name you passed to `vllm serve`.
    BASE_URL   = "http://localhost:8000"
    MODEL_NAME = "qwen3-vl-4b"

    # ── Inference settings ─────────────────────────────────────────────────
    TEMPERATURE = 0.0
    MAX_TOKENS  = 512

    # ── Parallelism / retry ────────────────────────────────────────────────
    # Set num_processes to roughly match vLLM's --max-num-seqs for best throughput.
    NUM_PROCESSES   = 10
    TIMEOUT_SECONDS = 60
    ENABLE_RETRY    = True
    MAX_RETRIES     = 1
    RETRY_DELAY     = 1.0

    # ══════════════════════════════════════════════════════════════════════

    if TEST_MODE:
        print("\n" + "=" * 80)
        print("TEST MODE – single image")
        print("=" * 80 + "\n")

        imgs = sorted(Path(IMAGES_FOLDER).glob("*"))
        imgs = [f for f in imgs if f.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        if not imgs:
            print(f"No images found in {IMAGES_FOLDER}")
            sys.exit(1)

        print(f"Testing with: {imgs[0].name}\n")
        ocr = Qwen3VLvLLMOCR(base_url=BASE_URL, model_name=MODEL_NAME, verbose=True)
        try:
            result = ocr.extract_text(
                image_path=str(imgs[0]),
                prompt=third_prompt,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                timeout=TIMEOUT_SECONDS,
            )
            print(f"Extracted text:\n{result['text']}\n")
            print("Test passed. Set TEST_MODE = False to run on all images.")
        except Exception as e:
            print(f"Test failed: {e}")
            print(f"\nEndpoint : {ocr.endpoint}")
            print("Check    : vLLM server running? Model name correct?")
            print(f"Verify   : curl {BASE_URL}/v1/models")
            traceback.print_exc()
            sys.exit(1)
        sys.exit(0)

    try:
        predict_vllm(
            images_folder=IMAGES_FOLDER,
            output_csv=OUTPUT_CSV,
            log_file=LOG_FILE,
            prompt=third_prompt,
            base_url=BASE_URL,
            model_name=MODEL_NAME,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            timeout_seconds=TIMEOUT_SECONDS,
            enable_retry=ENABLE_RETRY,
            max_retries=MAX_RETRIES,
            retry_delay=RETRY_DELAY,
            num_processes=NUM_PROCESSES,
        )
        print("\nProcessing completed successfully!")
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        traceback.print_exc()
        sys.exit(1)
