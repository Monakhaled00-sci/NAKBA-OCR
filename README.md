# NAKBA NLP 2026 – Shared Task 2 (Automation Track)

**Team:** Not Gemma  
**Task:** Arabic Handwritten Manuscript OCR  
**Model:** Qwen3-VL-4B-Instruct (full fine-tune, 32 epochs)

---

## Results

| Split | CER | WER |
|-------|----:|----:|
| Test set | 8.59 % | 25.87 % |
| Blind set | 0.11 % | 0.31 % |

---

## System Description

We fine-tuned **Qwen3-VL-4B-Instruct** end-to-end on the NAKBA 2026 training data using the HuggingFace `Trainer`. The model receives a single-cell image together with an English instruction prompt and is trained to predict the transcribed Arabic text directly. No post-processing or ensemble methods were used.

Key design choices:
- **Full fine-tuning** (all weights updated) – no LoRA / QLoRA
- **32 training epochs** with cosine LR schedule and 10 % warm-up
- **BF16** precision + gradient checkpointing for memory efficiency
- **Labels masked** on the prompt tokens – the loss is computed only on the assistant response
- Two inference paths: **HuggingFace** (single GPU, sequential) and **vLLM** (parallel HTTP, faster throughput)
- All training and inference ran on an **NVIDIA H100 (80 GB) server**

---

## Repository Structure

```
.
├── train.py             # Fine-tuning (HuggingFace Trainer)
├── evaluate.py          # Test-set inference + CER/WER metrics (HuggingFace)
├── predict.py           # Blind-set inference → submission CSV (HuggingFace)
├── predict_vllm.py      # Blind-set inference → submission CSV (vLLM, parallel)
├── prompts.py           # OCR prompts used in all runs
├── requirements.txt     # Python dependencies
└── data/                # NOT included – see Data Format below
    ├── train/
    │   ├── images/
    │   └── annotations.csv
    ├── test/
    │   ├── images/
    │   └── annotations.csv
    └── blind/
        └── images/
```

---

## Data Format

Each CSV annotation file must have exactly two columns:

| image | text |
|-------|------|
| page_001.jpg | النص العربي المكتوب |
| page_002.jpg | ... |

`image` is the bare filename (no path); `text` is the ground-truth Arabic transcript.

---

## Installation

### 1 – Clone the repository

```bash
git clone https://github.com/Monakhaled00-sci/NAKBA-OCR.git
cd NAKBA-OCR
```

### 2 – Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows
```

### 3 – Install dependencies

```bash
pip install -r requirements.txt
```

> **GPU requirement:**
> - Fine-tuning: H100 or A100 (80 GB VRAM) recommended for full fine-tuning.
> - HuggingFace inference: any GPU with ≥ 24 GB VRAM.
> - vLLM inference: same GPU requirement; vLLM must be installed separately (see below).
>
> All experiments in this work were conducted on an **NVIDIA H100 (80 GB) server**.

### 4 – Download the base model

```bash
huggingface-cli download Qwen/Qwen3-VL-4B-Instruct \
    --local-dir ./models/Qwen3-VL-4B-Instruct
```

Or set `model_path` in the scripts to the HF hub ID `"Qwen/Qwen3-VL-4B-Instruct"` and transformers will download it automatically.

---

## Reproducing the Results

### Step 1 – Fine-tune

Open `train.py` and set the paths in the `CONFIG` block at the bottom:

```python
config = FinetuneConfig(
    model_path            = "./models/Qwen3-VL-4B-Instruct",
    output_dir            = "./output/qwen3-vl-4b-arabic-ocr",
    train_images_dir      = "./data/train/images",
    train_annotations_csv = "./data/train/annotations.csv",
    val_images_dir        = "./data/val/images",
    val_annotations_csv   = "./data/val/annotations.csv",
    num_epochs            = 32,
    ...
)
```

Then run:

```bash
python train.py
```

The fine-tuned model is saved to `<output_dir>/final_model`.

Monitor training with TensorBoard:

```bash
tensorboard --logdir ./output/qwen3-vl-4b-arabic-ocr/tensorboard
```

---

### Step 2 – Evaluate on the test set

Open `evaluate.py`, update the paths in the `CONFIG` block at the bottom, then run:

```bash
python evaluate.py
```

This writes a JSON file with per-image CER / WER and an aggregate summary entry at the end.

---

### Step 3a – Blind-set submission (HuggingFace, sequential)

Open `predict.py`, update the paths, then run:

```bash
python predict.py
```

---

### Step 3b – Blind-set submission (vLLM, parallel – faster)

This approach spawns multiple HTTP workers that send requests to a vLLM server concurrently, giving significantly higher throughput than sequential HuggingFace inference.

#### Serve the model with vLLM

Install vLLM (separate from the main requirements):

```bash
pip install vllm>=0.8.0
```

Start the vLLM server pointing at the fine-tuned model:

```bash
vllm serve ./output/qwen3-vl-4b-arabic-ocr/final_model \
    --port 7834 \
    --quantization fp8 \
    --gpu-memory-utilization 0.3 \
    --max-model-len 1024 \
    --trust-remote-code
```

Key flags explained:

| Flag | Purpose |
|------|---------|
| `--port 7834` | Port the server listens on – must match `BASE_URL` in `predict_vllm.py` |
| `--quantization fp8` | FP8 quantization to reduce VRAM usage |
| `--gpu-memory-utilization 0.3` | Fraction of GPU memory allocated to vLLM (30 %) |
| `--max-model-len 1024` | Maximum context length in tokens |
| `--trust-remote-code` | Required for Qwen3-VL custom architecture |

Verify the server is up:

```bash
curl http://localhost:7834/v1/models
```

You should see `qwen3-vl-4b` in the response.

#### Run parallel inference

Open `predict_vllm.py`, set the paths and server details in the `CONFIG` block:

```python
BASE_URL      = "http://localhost:7834"   # must match --host / --port above
MODEL_NAME    = "qwen3-vl-4b"            # must match --served-model-name above
NUM_PROCESSES = 10                        # parallel HTTP workers
```

Then run:

```bash
python predict_vllm.py
```

> **Tip:** Set `TEST_MODE = True` first to verify the server is reachable with a single image before processing the full blind set.

---

## Prompt

The `third_prompt` in `prompts.py` was used for all final runs:

```
Transcribe all Arabic handwritten text from this image exactly as written,
preserving diacritics, punctuation, and layout. Output only the transcribed text.
```

---

## Team & Collaborators

- Mona Khaled
- Ali Adel
- Mohamed Emad
- Ibrahim Nasser