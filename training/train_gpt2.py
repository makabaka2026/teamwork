"""GPT-2 诗词微调训练"""
import argparse
from pathlib import Path

import torch
from transformers import (
    GPT2LMHeadModel, BertTokenizer,
    Trainer, TrainingArguments,
    TextDataset, DataCollatorForLanguageModeling,
)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] 训练设备: {device}")
    print(f"[INFO] 基础模型: {args.model_name}")

    # 加载 tokenizer 和模型
    tokenizer = BertTokenizer.from_pretrained(args.model_name)
    model = GPT2LMHeadModel.from_pretrained(args.model_name)
    model.to(device)

    # 准备数据集
    data_path = Path(args.data_dir) / "tang_gpt2.jsonl"
    if not data_path.exists():
        raise FileNotFoundError(f"训练数据不存在: {data_path}")

    # 转换为 HuggingFace TextDataset
    train_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=str(data_path),
        block_size=128,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # GPT-2 是因果语言模型
    )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        save_steps=500,
        save_total_limit=2,
        logging_steps=100,
        learning_rate=args.lr,
        warmup_steps=200,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )

    print("[INFO] 开始训练...")
    trainer.train()

    # 保存
    model.save_pretrained(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))
    print(f"[INFO] 模型已保存: {output_dir / 'final'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="uer/gpt2-chinese-poem")
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--output", default="output/gpt2-poem")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    args = parser.parse_args()

    print("=" * 50)
    print("  GPT-2 诗词模型微调")
    print("=" * 50)
    train(args)


if __name__ == "__main__":
    main()
