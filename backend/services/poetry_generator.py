"""诗词生成核心 — GPT-2 / LSTM"""
import json
import random
import pickle
import torch
from pathlib import Path
from backend.config import Config


class PoetryGenerator:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_type = "none"
        self.gpu_available = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.gpu_available else "cpu")
        self._fallback = None
        self._lstm_trained = False

    @property
    def fallback(self) -> dict:
        """懒加载回退诗库"""
        if self._fallback is None:
            fallback_path = Config.CHECKPOINT_DIR / "fallback_poems.json"
            if fallback_path.exists():
                with open(fallback_path, encoding="utf-8") as f:
                    self._fallback = json.load(f)
            else:
                self._fallback = {}
        return self._fallback

    def load_model(self, model_type: str = "gpt2"):
        if model_type == "gpt2":
            try:
                self._load_gpt2()
                self.model_type = "gpt2"
            except Exception as e:
                print(f"[WARN] GPT-2 加载失败 ({e})，回退到 LSTM")
                self._load_lstm()
                self.model_type = "lstm"
        elif model_type == "lstm":
            self._load_lstm()
            self.model_type = "lstm"
        else:
            raise ValueError(f"未知模型类型: {model_type}")

    def _load_gpt2(self):
        from transformers import GPT2LMHeadModel, BertTokenizer

        model_name = "uer/gpt2-chinese-poem"
        print(f"[INFO] 加载 GPT-2 模型: {model_name} ...")

        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        print(f"[INFO] GPT-2 加载完成 (device={self.device})")

    def _load_lstm(self):
        from backend.services.lstm_model import PoemLSTM

        print("[INFO] 初始化 LSTM 模型 ...")

        # 尝试加载词汇表
        vocab_path = Config.CHECKPOINT_DIR / "vocab.pkl"
        word2ix, ix2word = {}, {}
        if vocab_path.exists():
            with open(vocab_path, "rb") as f:
                vocab = pickle.load(f)
            word2ix = vocab.get("word2ix", {})
            ix2word = vocab.get("ix2word", {})

        vocab_size = len(word2ix) if word2ix else 8000
        self.model = PoemLSTM(
            vocab_size=vocab_size,
            embedding_dim=256,
            hidden_dim=512,
            num_layers=3,
        )
        self.model.to(self.device)
        self.model.eval()

        # 注入词汇表
        self.model.word2ix = word2ix
        self.model.ix2word = ix2word

        # 尝试加载预训练权重
        ckpt_path = Config.LSTM_MODEL_PATH
        if ckpt_path.exists():
            print(f"[INFO] 加载 LSTM 权重: {ckpt_path}")
            self.model.load_state_dict(
                torch.load(ckpt_path, map_location=self.device, weights_only=True)
            )
            self._lstm_trained = True
        else:
            print("[WARN] 未找到 LSTM 权重，使用回退诗库")
            self._lstm_trained = False

        self.tokenizer = None
        print(f"[INFO] LSTM 加载完成 (device={self.device}, vocab={vocab_size})")

    def switch_model(self, model_type: str):
        if self.model_type != model_type:
            self.load_model(model_type)

    def generate(self, prompt: str, num_lines: int = 4, temperature: float = 0.8) -> str:
        if self.model is None:
            raise RuntimeError("模型未加载，请先调用 load_model()")

        if self.model_type == "gpt2":
            return self._generate_gpt2(prompt, num_lines, temperature)
        else:
            result = self._generate_lstm(prompt, num_lines, temperature)
            if result is None:
                return self._fallback_generate(prompt, num_lines)
            return result

    def _generate_gpt2(self, prompt: str, num_lines: int, temperature: float) -> str:
        max_len_per_line = 10
        max_length = len(prompt) + num_lines * max_len_per_line

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                max_length=max_length,
                temperature=temperature if temperature > 0 else 1.0,
                do_sample=temperature > 0,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.2,
                pad_token_id=self.tokenizer.pad_token_id or 0,
                eos_token_id=self.tokenizer.sep_token_id,
            )

        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        if generated.startswith(prompt):
            generated = generated[len(prompt):]
        return generated.strip()

    def _generate_lstm(self, prompt: str, num_lines: int, temperature: float) -> str | None:
        if not self._lstm_trained:
            return None
        if not hasattr(self.model, "word2ix") or not self.model.ix2word:
            return None
        if not self.model.word2ix:
            return None
        try:
            return self.model.generate(
                start_text=prompt,
                max_chars=num_lines * 10,
                temperature=temperature,
                device=self.device,
            )
        except Exception:
            return None

    def _fallback_generate(self, prompt: str, num_lines: int) -> str:
        """从回退诗库中按关键词匹配诗词"""
        if not self.fallback:
            return "春眠不觉晓，\n处处闻啼鸟。\n夜来风雨声，\n花落知多少。"

        # 在 prompt 中查找匹配的关键词
        candidates = []
        for kw, poems in self.fallback.items():
            if kw in prompt:
                candidates.extend(poems)

        if not candidates:
            # 随机返回一首
            all_poems = []
            for poems in self.fallback.values():
                all_poems.extend(poems)
            if all_poems:
                candidates = all_poems

        if not candidates:
            return "春眠不觉晓，\n处处闻啼鸟。\n夜来风雨声，\n花落知多少。"

        # 随机选一首，截取所需行数
        chosen = random.choice(candidates)
        lines = chosen[:num_lines]
        return "\n".join(lines)
