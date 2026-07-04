"""GPT-2 诗词微调训练 — 支持唐诗/宋词分模型训练"""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import (
    GPT2LMHeadModel, BertTokenizer,
    Trainer, TrainingArguments,
    DataCollatorForLanguageModeling,
)


class PoetryDataset(Dataset):
    """从 JSONL 文件加载诗词数据"""

    def __init__(self, file_path: str, tokenizer, block_size: int = 128):
        self.examples = []
        with open(file_path, encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                text = item.get("text", "")
                if text.strip():
                    enc = tokenizer(
                        text, truncation=True, max_length=block_size,
                        padding="max_length", return_tensors="pt",
                    )
                    self.examples.append(enc["input_ids"].squeeze(0))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return {"input_ids": self.examples[i], "labels": self.examples[i]}


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] 设备: {device}")
    print(f"[INFO] 模型: {args.model_name}")
    print(f"[INFO] 数据: {args.data_file}")

    # 加载 tokenizer 和模型
    tokenizer = BertTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(args.model_name).to(device)

    # 加载数据集
    data_path = Path(args.data_dir) / args.data_file
    if not data_path.exists():
        raise FileNotFoundError(f"数据文件不存在: {data_path}")

    dataset = PoetryDataset(str(data_path), tokenizer, block_size=128)
    print(f"[INFO] 样本数: {len(dataset)}")

    # 输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        save_steps=500,
        save_total_limit=2,
        logging_steps=50,
        learning_rate=args.lr,
        warmup_steps=200,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,
        save_strategy="steps",
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )

    print("[INFO] 开始训练...")
    trainer.train()

    # 保存最终模型
    final_dir = output_dir / "final"
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"[INFO] 模型已保存: {final_dir}")


def main():
    parser = argparse.ArgumentParser(description="GPT-2 诗词模型训练")
    parser.add_argument("--model_name", default="uer/gpt2-chinese-poem")
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--data_file", default="tang_gpt2.jsonl",
                        help="训练数据文件: tang_wuyan_gpt2.jsonl / tang_qiyan_gpt2.jsonl / songci_gpt2.jsonl")
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
