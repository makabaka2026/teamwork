"""诗词生成器 — BART Encoder-Decoder"""
import torch
from pathlib import Path
from backend.config import Config


class PoetryGenerator:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.style = "tang"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self, style="tang"):
        """加载 BART 模型"""
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.style = style
        model_dir = Config.get_model_dir(style)

        if not model_dir.exists():
            raise FileNotFoundError(
                f"模型不存在: {model_dir}\n"
                f"请先训练: python3 train_bart.py --data_file seq2seq_tang.jsonl"
            )

        print(f"[INFO] 加载模型: {model_dir}")
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self.model = AutoModelForSeq2SeqLM.from_pretrained(str(model_dir))
        self.model.to(self.device)
        self.model.eval()
        print(f"[INFO] 加载完成 (style={style}, device={self.device})")

    def generate(self, keyword, num_lines=4, temperature=0.8):
        """
        输入关键词 → 输出诗句
        Encoder 编码关键词, Decoder 生成诗句
        """
        if self.model is None:
            raise RuntimeError("模型未加载，请先调用 load_model()")

        inputs = self.tokenizer(keyword, return_tensors="pt").to(self.device)
        gen_kwargs = {k: v for k, v in inputs.items() if k != "token_type_ids"}

        with torch.no_grad():
            outputs = self.model.generate(
                **gen_kwargs,
                max_new_tokens=num_lines * 14,
                temperature=max(temperature, 0.01),
                do_sample=temperature > 0,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.2,
            )

        poem = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return poem.strip()
