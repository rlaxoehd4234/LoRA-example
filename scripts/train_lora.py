"""
Qwen2.5-1.5B-Instruct + LoRA SFT, tuned to fit inside 8GB of total RAM.

Config choices and why (see the "Qwen2.5 LoRA 구현 가이드" doc for the full
memory-budget breakdown):
  - rank=8, alpha=16, target_modules=[q_proj, v_proj] only
      -> keeps trainable params under ~1% of the base model, so LoRA itself
         barely adds to memory. Widen target_modules only if you have headroom.
  - batch_size=1 + gradient_accumulation_steps=4
      -> simulates an effective batch of 4 while only ever holding 1
         sequence's activations in memory at a time.
  - gradient_checkpointing=True
      -> NOT optional at this RAM budget. Recomputes activations during the
         backward pass instead of keeping them all in memory — trades some
         speed for a large memory cut.
  - max_length=384
      -> covers the median example in mock_data (~574 chars ≈ 350-450
         Qwen tokens) but truncates the long tail. That's the trade-off for
         staying inside 8GB — see the dataset stats in the guide doc.

Device policy:
  - CUDA available  -> bf16 LoRA (add --use-qlora for 4bit if you also have
                       bitsandbytes installed; not covered by this script's
                       default path to keep it portable)
  - MPS (Apple Silicon) -> fp32 LoRA. bitsandbytes doesn't support MPS
                       reliably, and fp16/bf16 matmul on MPS still has
                       occasional NaN issues on some transformers versions —
                       fp32 is the boring, safe choice here.
  - CPU only        -> fp32 LoRA, same config. Just slower (see timeline
                       in the guide doc: hours, not minutes).

Usage:
    SFT_MAX_LENGTH=384 SFT_EPOCHS=2 SFT_LEARNING_RATE=1e-4 SFT_GRAD_ACCUM=4 \
    python scripts/train_lora.py
"""
import json
import os
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = os.environ.get("SFT_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
ADAPTER_OUT = ROOT / "models" / "hr-qwen-lora"
OUTPUTS_DIR = ROOT / "outputs"

MAX_LENGTH = int(os.environ.get("SFT_MAX_LENGTH", 384))
EPOCHS = float(os.environ.get("SFT_EPOCHS", 2))
LEARNING_RATE = float(os.environ.get("SFT_LEARNING_RATE", 1e-4))
GRAD_ACCUM = int(os.environ.get("SFT_GRAD_ACCUM", 4))
LORA_RANK = int(os.environ.get("SFT_LORA_RANK", 8))


def detect_device_type() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def validate_messages(example: dict) -> bool:
    roles = [m["role"] for m in example["messages"]]
    return roles == ["system", "user", "assistant"] and all(
        m["content"].strip() for m in example["messages"]
    )


def load_training_dataset():
    data_files = {
        "train": str(ROOT / "mock_data" / "hr_sft_train.jsonl"),
        "validation": str(ROOT / "mock_data" / "hr_sft_validation.jsonl"),
    }
    ds = load_dataset("json", data_files=data_files)
    for split in ds:
        bad = [i for i, ex in enumerate(ds[split]) if not validate_messages(ex)]
        if bad:
            raise ValueError(f"{split}: malformed examples at rows {bad}")
    return ds


def main():
    device_type = detect_device_type()
    print(f"[train_lora] device={device_type} rank={LORA_RANK} max_length={MAX_LENGTH} "
          f"epochs={EPOCHS} lr={LEARNING_RATE} grad_accum={GRAD_ACCUM}")

    # fp32 on MPS/CPU for stability; bf16 on CUDA to roughly halve weight memory.
    dtype = torch.bfloat16 if device_type == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
        device_map={"": device_type} if device_type != "cpu" else None,
    )
    model.config.use_cache = False  # required alongside gradient checkpointing

    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_RANK * 2,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = load_training_dataset()

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    ADAPTER_OUT.mkdir(parents=True, exist_ok=True)

    sft_config = SFTConfig(
        output_dir=str(OUTPUTS_DIR / "checkpoints"),
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        gradient_checkpointing=True,          # mandatory at this RAM budget — see module docstring
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=MAX_LENGTH,
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="no",                    # we save the final adapter ourselves below
        report_to=[],
        bf16=(device_type == "cuda"),
        assistant_only_loss=True,              # only train on the assistant's turn, not the prompt
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
    )

    train_result = trainer.train()

    model.save_pretrained(str(ADAPTER_OUT))
    tokenizer.save_pretrained(str(ADAPTER_OUT))

    log_history = trainer.state.log_history
    (OUTPUTS_DIR / "training_log.json").write_text(
        json.dumps(log_history, indent=2, ensure_ascii=False)
    )
    summary = {
        "model": MODEL_NAME,
        "device": device_type,
        "lora_rank": LORA_RANK,
        "epochs": EPOCHS,
        "max_length": MAX_LENGTH,
        "train_examples": len(dataset["train"]),
        "final_train_loss": train_result.training_loss,
        "adapter_path": str(ADAPTER_OUT),
    }
    (OUTPUTS_DIR / "training_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )
    print(f"[train_lora] done. adapter -> {ADAPTER_OUT}")
    print(f"[train_lora] summary -> {OUTPUTS_DIR / 'training_summary.json'}")


if __name__ == "__main__":
    main()
