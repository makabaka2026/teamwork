"""BART Seq2Seq 训练 — 关键词 → 诗句"""
import os, argparse, json
from pathlib import Path
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSeq2SeqLM, AutoTokenizer,
    Seq2SeqTrainingArguments, Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)

# 国内优先用镜像
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

MODEL_NAME = "fnlp/bart-base-chinese"


class PoemSeq2SeqDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_src=32, max_tgt=128):
        self.tokenizer = tokenizer
        self.max_src = max_src
        self.max_tgt = max_tgt
        self.pairs = []
        with open(file_path, encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                src = item.get("input", "")
                tgt = item.get("output", "")
                if src.strip() and tgt.strip():
                    self.pairs.append((src, tgt))

    def __len__(self): return len(self.pairs)

    def __getitem__(self, i):
        src, tgt = self.pairs[i]
        src_enc = self.tokenizer(src, truncation=True, max_length=self.max_src,
                                  padding="max_length", return_tensors="pt")
        tgt_enc = self.tokenizer(tgt, truncation=True, max_length=self.max_tgt,
                                  padding="max_length", return_tensors="pt")
        return {
            "input_ids": src_enc["input_ids"].squeeze(0),
            "attention_mask": src_enc["attention_mask"].squeeze(0),
            "labels": tgt_enc["input_ids"].squeeze(0),
        }


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] 设备: {device}")
    print(f"[INFO] 模型: {MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)

    data_path = Path(args.data_dir) / args.data_file
    dataset = PoemSeq2SeqDataset(str(data_path), tokenizer)
    print(f"[INFO] 训练对: {len(dataset)}")

    split = int(len(dataset) * 0.95)
    train_ds = torch.utils.data.Subset(dataset, range(split))
    val_ds = torch.utils.data.Subset(dataset, range(split, len(dataset)))

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_steps=200,
        weight_decay=0.01,
        logging_steps=50,
        save_steps=500,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        predict_with_generate=True,
        generation_max_length=128,
        eval_strategy="steps",
        eval_steps=500,
        dataloader_num_workers=0,
    )

    trainer = Seq2SeqTrainer(
        model=model, args=training_args,
        train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    )

    print("[INFO] 开始训练...")
    trainer.train()

    final_dir = output_dir / "final"
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"[INFO] 模型已保存: {final_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--data_file", default="seq2seq_tang.jsonl")
    parser.add_argument("--output", default="output/bart-tang")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    args = parser.parse_args()
    print("=" * 50)
    print("  BART 诗词 Seq2Seq 训练")
    print("=" * 50)
    train(args)


if __name__ == "__main__":
    main()
