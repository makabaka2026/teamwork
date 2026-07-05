"""BART 模型评估"""
import re, argparse
from pathlib import Path
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="output/bart-tang/final")
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_dir = Path(args.model_dir)

    if not model_dir.exists():
        print(f"[ERROR] 模型不存在: {model_dir}")
        return

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSeq2SeqLM.from_pretrained(str(model_dir)).to(device)
    model.eval()

    print("=" * 55)
    print(f"  BART 诗词生成 ({args.model_dir})")
    print("=" * 55)

    tests = ["春风", "明月", "边塞", "离别", "江南", "相思", "秋日", "登高"]

    for kw in tests:
        inputs = tokenizer(kw, return_tensors="pt").to(device)
        # BART 不需要 token_type_ids
        gen_kwargs = {k: v for k, v in inputs.items() if k != "token_type_ids"}
        with torch.no_grad():
            outputs = model.generate(**gen_kwargs, max_new_tokens=56,
                temperature=args.temperature, do_sample=True,
                top_p=0.9, top_k=50, repetition_penalty=1.2)
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        lines = re.split(r"[，。！？、；：\n]+", text)
        lines = [l.strip() for l in lines if l.strip() and
                 len(re.findall(r"[一-鿿]", l)) >= 4][:4]
        lens = [len(re.findall(r"[一-鿿]", l)) for l in lines]

        print(f"\n  【{kw}】  { '/'.join(str(l) for l in lens) }")
        for l in lines[:4]:
            print(f"    {l}")


if __name__ == "__main__":
    main()
