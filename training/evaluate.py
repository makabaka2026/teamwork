"""模型评估 — BLEU / Perplexity / 人工评估辅助"""
import argparse
from pathlib import Path

import torch
import numpy as np
from transformers import GPT2LMHeadModel, BertTokenizer


def calc_perplexity(model, tokenizer, texts: list[str], device) -> float:
    """计算困惑度 Perplexity"""
    import math
    total_loss = 0
    total_tokens = 0

    model.eval()
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt").to(device)
            outputs = model(**inputs, labels=inputs["input_ids"])
            total_loss += outputs.loss.item() * inputs["input_ids"].size(1)
            total_tokens += inputs["input_ids"].size(1)

    if total_tokens == 0:
        return float("inf")
    return math.exp(total_loss / total_tokens)


def calc_bleu(references: list[list[str]], candidates: list[str]) -> float:
    """计算 BLEU 分数（使用 sacrebleu）"""
    try:
        from sacrebleu import corpus_bleu
        score = corpus_bleu(candidates, references)
        return score.score
    except ImportError:
        print("[WARN] sacrebleu 未安装，跳过 BLEU 计算")
        return -1


def manual_eval_sample(model, tokenizer, prompts: list[str], device) -> list[dict]:
    """生成样本供人工评估"""
    results = []
    model.eval()
    with torch.no_grad():
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            outputs = model.generate(
                inputs["input_ids"],
                max_length=len(prompt) + 60,
                temperature=0.8,
                do_sample=True,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.2,
            )
            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            results.append({"prompt": prompt, "generated": generated})
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="output/gpt2-poem/final")
    parser.add_argument("--test_data", default="data/processed/tang_gpt2.jsonl")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = Path(args.model_dir)

    if not model_dir.exists():
        print(f"[ERROR] 模型目录不存在: {model_dir}")
        return

    print("[INFO] 加载模型...")
    model = GPT2LMHeadModel.from_pretrained(str(model_dir)).to(device)
    tokenizer = BertTokenizer.from_pretrained(str(model_dir))

    # 测试样本
    test_prompts = [
        "春",
        "明月",
        "离别",
        "边塞",
        "山水",
    ]

    print("\n" + "=" * 50)
    print("  生成样本")
    print("=" * 50)
    samples = manual_eval_sample(model, tokenizer, test_prompts, device)
    for s in samples:
        print(f"\n  Prompt: {s['prompt']}")
        print(f"  Generated:\n    {s['generated']}")
        print("  " + "-" * 40)


if __name__ == "__main__":
    main()
