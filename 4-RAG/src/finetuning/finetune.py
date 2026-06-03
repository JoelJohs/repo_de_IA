#!/usr/bin/env python3
"""
Fine-tuning LoRA sobre Qwen2.5-0.5B-Instruct usando el dataset generado.
Requiere: pip install torch transformers peft datasets accelerate

Uso:  python src/finetuning/finetune.py --dataset src/finetuning/dataset/rag_dataset.jsonl

Advertencia: El entrenamiento en CPU es extremadamente lento.
Para uso real, se recomienda GPU.
"""

import json, sys, argparse, os
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training,
)

BASE_DIR = Path(__file__).parent.parent.parent
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_DIR = BASE_DIR / "src" / "finetuning" / "lora_adapter"
DEFAULT_DATASET = BASE_DIR / "datasets" / "finetuning.jsonl"


def load_dataset_jsonl(filepath):
    records = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def format_example(record):
    instruction = record.get("instruction", "")
    output = record.get("response") or record.get("output", "")
    text = f"<|system|>\nEres un asistente experto en seguridad pública en México. Responde de manera útil, precisa y basada en fuentes.</s>\n<|user|>\n{instruction}</s>\n<|assistant|>\n{output}</s>"
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=str(DEFAULT_DATASET))
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--no-cuda", action="store_true", help="Force CPU")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    print(f"  Device: {device}")
    if device == "cpu":
        print("  WARNING: CPU training is extremely slow. Consider using --no-cuda only for testing.")
    else:
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    print(f"\n  Loading dataset: {args.dataset}")
    records = load_dataset_jsonl(args.dataset)
    print(f"  Records: {len(records)}")

    texts = [format_example(r) for r in records]
    dataset = Dataset.from_list([{"text": t} for t in texts])

    print(f"\n  Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        local_files_only=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32 if device == "cpu" else torch.bfloat16,
        device_map={"": device} if device == "cuda" else None,
        trust_remote_code=True,
        local_files_only=False,
    )

    print(f"  Applying LoRA (r={args.lora_r}, alpha={args.lora_alpha})")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=args.max_length,
        )

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        warmup_steps=10,
        logging_steps=5,
        save_steps=50,
        save_total_limit=2,
        fp16=False,
        bf16=False,
        dataloader_pin_memory=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
    )

    print(f"\n  Starting training ({args.epochs} epochs)...")
    trainer.train()

    print(f"\n  Saving LoRA adapter to {args.output_dir}")
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))

    print("  Done.")


if __name__ == "__main__":
    main()
