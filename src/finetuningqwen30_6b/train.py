"""LoRA 微调 Qwen3-0.6B，让模型模仿甄嬛的说话风格"""
import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

BASE_MODEL = "Qwen/Qwen3-0.6B"
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = PROJECT_ROOT / "datasets" / "huanhuan_train.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "model" / "qwen3-0.6b-huanhuan-lora"

SEED = 42


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--max_steps", type=int, default=-1, help=">0 时覆盖 num_train_epochs，用于快速试跑")
    return parser.parse_args()


def main():
    args = parse_args()
    print("训练开始")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    # 因果语言模型训练需要 pad，Qwen3 默认没有单独设置 pad_token
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16)
    model.config.use_cache = False

    dataset = load_dataset("json", data_files=str(DATA_PATH), split="train")
    split = dataset.train_test_split(test_size=0.05, seed=SEED)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    print(f"训练集: {len(train_dataset)} 条，验证集: {len(eval_dataset)} 条")

    lora_config = LoraConfig(
        r=16,  # 低秩矩阵的秩，控制可训练参数量
        lora_alpha=32,  # 缩放系数，通常取 r 的两倍
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,  # 等效批大小 = 8 * 2 = 16
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=20,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        max_length=512,
        packing=False,
        assistant_only_loss=True,
        dataset_num_proc=1,
        report_to="tensorboard",
        seed=SEED,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"LoRA 适配器已保存至: {args.output_dir}")


if __name__ == "__main__":
    main()
