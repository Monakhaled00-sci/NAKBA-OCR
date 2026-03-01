"""
Fine-tuning script for Qwen3-VL on Arabic handwritten OCR.

Usage
-----
Edit the CONFIG block at the bottom of this file, then run:

    python train.py

All paths, hyperparameters, and prompts are controlled from that single block.
"""

import os
import json
import logging
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from functools import partial

import torch
import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)
from qwen_vl_utils import process_vision_info
from prompts import third_prompt

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class FinetuneConfig:
    """All fine-tuning settings in one place."""

    # ── Required paths ────────────────────────────────────────────────────────
    model_path: str = ""          # Path to base or previously fine-tuned model
    output_dir: str = ""          # Where checkpoints and the final model are saved
    train_images_dir: str = ""    # Directory of training images
    train_annotations_csv: str = "" # CSV with columns: image, text

    # ── Optional validation paths (if omitted, 10 % of train is used) ────────
    val_images_dir: Optional[str] = None
    val_annotations_csv: Optional[str] = None

    # ── Hyperparameters ───────────────────────────────────────────────────────
    num_epochs: int = 32
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_length: int = 512          # Max token length for inputs

    # ── Evaluation / saving ───────────────────────────────────────────────────
    eval_steps: int = 200
    save_steps: int = 400          # Must be a multiple of eval_steps
    save_total_limit: int = 3
    logging_steps: int = 10

    # ── System ────────────────────────────────────────────────────────────────
    dataloader_num_workers: int = 4
    seed: int = 42
    bf16: bool = True
    gradient_checkpointing: bool = True

    # ── Prompts ───────────────────────────────────────────────────────────────
    system_prompt: str = ""
    user_prompt: str = field(default_factory=lambda: third_prompt)


# ── Dataset ───────────────────────────────────────────────────────────────────

class ArabicOCRDataset(Dataset):
    """Dataset for Arabic OCR fine-tuning with Qwen3-VL."""

    def __init__(
        self,
        annotations_csv: str,
        images_dir: str,
        processor: AutoProcessor,
        config: FinetuneConfig,
    ):
        self.images_dir = images_dir
        self.processor = processor
        self.config = config

        self.df = pd.read_csv(annotations_csv)
        logger.info(f"Loaded {len(self.df)} samples from {annotations_csv}")
        self._filter_missing_images()

    def _filter_missing_images(self):
        missing = [
            img for img in self.df["image"]
            if not os.path.exists(os.path.join(self.images_dir, img))
        ]
        if missing:
            logger.warning(f"Missing {len(missing)} images – they will be skipped.")
            self.df = self.df[~self.df["image"].isin(missing)].reset_index(drop=True)
        logger.info(f"Dataset size after filtering: {len(self.df)}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = os.path.join(self.images_dir, row["image"])
        text = str(row["text"])

        messages = [
            {"role": "system", "content": self.config.system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": self.config.user_prompt},
                ],
            },
            {"role": "assistant", "content": text},
        ]
        return {"messages": messages}


# ── Data collator ─────────────────────────────────────────────────────────────

def collate_fn(batch: List[Dict], processor: AutoProcessor, config: FinetuneConfig):
    """Collate a batch of conversations into model inputs, masking prompt tokens."""

    texts, all_images = [], []
    for item in batch:
        messages = item["messages"]
        texts.append(
            processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        )
        imgs, _ = process_vision_info(messages)
        if imgs:
            all_images.extend(imgs)

    proc_kwargs = dict(
        text=texts,
        padding=True,
        truncation=True,
        max_length=config.max_length,
        return_tensors="pt",
    )
    if all_images:
        proc_kwargs["images"] = all_images

    inputs = processor(**proc_kwargs)

    labels = inputs["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100

    # Mask everything before the assistant response
    im_start_id = processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
    assistant_id = processor.tokenizer.encode("assistant", add_special_tokens=False)[0]

    for i, ids in enumerate(inputs["input_ids"].tolist()):
        last_assistant_pos = -1
        for j in range(len(ids) - 1):
            if ids[j] == im_start_id and ids[j + 1] == assistant_id:
                last_assistant_pos = j
        if last_assistant_pos != -1:
            labels[i, : last_assistant_pos + 3] = -100  # mask <|im_start|> + "assistant" + \n

    inputs["labels"] = labels
    return inputs


# ── Callbacks ─────────────────────────────────────────────────────────────────

class LoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and state.is_world_process_zero:
            step = state.global_step
            if "loss" in logs:
                logger.info(f"Step {step}: loss={logs['loss']:.4f}")
            if "eval_loss" in logs:
                logger.info(f"Step {step}: eval_loss={logs['eval_loss']:.4f}")


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model_and_processor(config: FinetuneConfig):
    logger.info(f"Loading model: {config.model_path}")

    processor = AutoProcessor.from_pretrained(
        config.model_path, trust_remote_code=True, local_files_only=True
    )
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    model = AutoModelForVision2Seq.from_pretrained(
        config.model_path,
        torch_dtype=torch.bfloat16 if config.bf16 else torch.float16,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
    )

    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled.")

    model.train()

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Parameters – total: {total:,}  trainable: {trainable:,}")

    return model, processor


# ── Dataset preparation ───────────────────────────────────────────────────────

def prepare_datasets(config: FinetuneConfig, processor: AutoProcessor):
    train_ds = ArabicOCRDataset(config.train_annotations_csv, config.train_images_dir, processor, config)

    if config.val_annotations_csv and config.val_images_dir:
        val_ds = ArabicOCRDataset(config.val_annotations_csv, config.val_images_dir, processor, config)
        logger.info(f"Using separate validation set ({len(val_ds)} samples).")
    else:
        logger.info("No validation set provided – using 10 % of training data.")
        train_df, val_df = train_test_split(train_ds.df, test_size=0.1, random_state=config.seed)
        train_ds.df = train_df.reset_index(drop=True)

        val_ds = ArabicOCRDataset(config.train_annotations_csv, config.train_images_dir, processor, config)
        val_ds.df = val_df.reset_index(drop=True)

    logger.info(f"Train: {len(train_ds)}  |  Val: {len(val_ds)}")
    return train_ds, val_ds


# ── Main training function ────────────────────────────────────────────────────

def run_training(config: FinetuneConfig):
    print("\n" + "=" * 70)
    print(" " * 15 + "QWEN3-VL ARABIC OCR – FINE-TUNING")
    print("=" * 70)

    os.makedirs(config.output_dir, exist_ok=True)

    # Log to file
    log_file = os.path.join(config.output_dir, "training.log")
    fh = logging.FileHandler(log_file)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)

    # Save config snapshot
    with open(os.path.join(config.output_dir, "config.json"), "w") as f:
        json.dump(vars(config), f, indent=2, ensure_ascii=False)

    model, processor = load_model_and_processor(config)
    train_ds, val_ds = prepare_datasets(config, processor)

    effective_batch = config.batch_size * config.gradient_accumulation_steps
    steps_per_epoch = max(len(train_ds) // effective_batch, 1)
    total_steps = steps_per_epoch * config.num_epochs

    logger.info(
        f"\nEffective batch size: {effective_batch} | "
        f"Steps/epoch: {steps_per_epoch} | "
        f"Total steps: {total_steps} | "
        f"LR: {config.learning_rate}"
    )

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        overwrite_output_dir=False,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        bf16=config.bf16,
        fp16=not config.bf16,
        logging_dir=os.path.join(config.output_dir, "tensorboard"),
        logging_steps=config.logging_steps,
        report_to=["tensorboard"],
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_num_workers=config.dataloader_num_workers,
        dataloader_pin_memory=True,
        seed=config.seed,
        gradient_checkpointing=config.gradient_checkpointing,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=partial(collate_fn, processor=processor, config=config),
        callbacks=[LoggingCallback()],
    )

    logger.info(f"Training started at {datetime.now():%Y-%m-%d %H:%M:%S}")
    trainer.train()

    final_path = os.path.join(config.output_dir, "final_model")
    trainer.save_model(final_path)
    processor.save_pretrained(final_path)
    logger.info(f"Final model saved to: {final_path}")

    with open(os.path.join(config.output_dir, "training_metrics.json"), "w") as f:
        json.dump(trainer.state.log_history, f, indent=2)

    print(f"\n{'='*70}")
    print(f"  TRAINING COMPLETE – model at: {final_path}")
    print(f"  TensorBoard: tensorboard --logdir={os.path.join(config.output_dir, 'tensorboard')}")
    print(f"{'='*70}\n")


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION – edit this block before running
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    config = FinetuneConfig(
        # ── Paths ──────────────────────────────────────────────────────────
        # Use the original Qwen3-VL-4B-Instruct for a fresh run, or point
        # to a previous fine-tune checkpoint to continue training.
        model_path="Qwen/Qwen3-VL-4B-Instruct",   # HF hub ID or local path

        output_dir="./output/qwen3-vl-4b-arabic-ocr",

        train_images_dir="./data/train/images",
        train_annotations_csv="./data/train/annotations.csv",

        # Leave as None to auto-split 10 % from training data
        val_images_dir="./data/val/images",
        val_annotations_csv="./data/val/annotations.csv",

        # ── Hyperparameters (values used in the competition) ───────────────
        num_epochs=32,
        batch_size=2,
        gradient_accumulation_steps=8,   # effective batch = 16
        learning_rate=2e-5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        max_length=512,

        # ── Checkpointing ──────────────────────────────────────────────────
        eval_steps=200,
        save_steps=400,   # must be a multiple of eval_steps
        save_total_limit=3,
        logging_steps=10,

        # ── System ─────────────────────────────────────────────────────────
        dataloader_num_workers=4,
        seed=42,
        bf16=True,
        gradient_checkpointing=True,
    )

    run_training(config)
