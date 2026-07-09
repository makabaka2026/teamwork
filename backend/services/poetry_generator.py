"""诗词生成核心 — BART / GPT-2 / LSTM（含格式约束解码）"""
import json
import random
import re
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
        # BART 多模型
        self._bart_models = {}  # style -> (tokenizer, model)

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

    def load_model(self, model_type: str = "bart"):
        if model_type == "bart":
            try:
                self._load_bart()
                self.model_type = "bart"
            except Exception as e:
                print(f"[WARN] BART 加载失败 ({e})，回退到 GPT-2")
                try:
                    self._load_gpt2()
                    self.model_type = "gpt2"
                except Exception as e2:
                    print(f"[WARN] GPT-2 加载失败 ({e2})，回退到 LSTM")
                    self._load_lstm()
                    self.model_type = "lstm"
        elif model_type == "gpt2":
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

    def _load_bart(self):
        """加载本地训练的 BART 模型（五言/七言/宋词）"""
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        print("[INFO] 加载本地 BART 模型...")
        model_dirs = {
            "wuyan": Config.BART_WUYAN_DIR,
            "qiyan": Config.BART_QIYAN_DIR,
            "song": Config.BART_SONG_DIR,
        }
        for style, path in model_dirs.items():
            if path.exists():
                tok = AutoTokenizer.from_pretrained(str(path))
                mdl = AutoModelForSeq2SeqLM.from_pretrained(str(path))
                mdl.to(self.device)
                mdl.eval()
                self._bart_models[style] = (tok, mdl)
                print(f"  [BART] {style}: {path}")

        if not self._bart_models:
            raise RuntimeError("无可用 BART 模型")

        # 设置默认 tokenizer/model (第一个可用)
        first = list(self._bart_models.keys())[0]
        self.tokenizer, self.model = self._bart_models[first]
        print(f"[INFO] BART 加载完成 (device={self.device}, styles={list(self._bart_models.keys())})")

    def _generate_bart(self, prompt: str, num_lines: int, temperature: float,
                       style: str = "wuyan") -> str:
        """BART 生成诗词 + 韵脚约束修正（用 embedding 相似度找最佳替换字）"""
        import time
        from backend.utils.rhyme_dict import get_final, get_rhyming_chars

        # 风格映射
        style_map = {"tang": "qiyan", "wuyan": "wuyan", "qiyan": "qiyan", "song": "song"}
        bart_style = style_map.get(style, "qiyan")

        if bart_style in self._bart_models:
            tok, mdl = self._bart_models[bart_style]
        else:
            tok, mdl = self.tokenizer, self.model

        expected_len = 7 if bart_style == "qiyan" else 5
        if bart_style == "song":
            # 宋词：自由生成，不强制韵脚
            action = "写一首词，押韵工整。"
            full_prompt = f"{action}{prompt}。"
            max_tokens = num_lines * 24
        else:
            yan = "七言" if bart_style == "qiyan" else "五言"
            form = "绝句" if num_lines <= 4 else ("律诗" if num_lines == 8 else "古诗")
            action = f"写一首{yan}{form}，偶句押韵。主题："
            full_prompt = f"{action}{prompt}。"
            max_tokens = num_lines * 18

        # === 第一阶段：自由生成（保留诗意） ===
        rhyme_temp = min(temperature, 0.7)
        torch.manual_seed(int(time.time() * 1000) % (2**31))
        inputs = tok(full_prompt, return_tensors="pt").to(self.device)
        gen_kwargs = {k: v for k, v in inputs.items() if k != "token_type_ids"}

        outputs = mdl.generate(
            **gen_kwargs,
            max_new_tokens=max_tokens,
            temperature=max(rhyme_temp, 0.1),
            do_sample=(rhyme_temp > 0),
            top_p=0.90,
            top_k=50,
            repetition_penalty=1.5,
            num_beams=1,
            no_repeat_ngram_size=2,
        )

        poem = tok.decode(outputs[0], skip_special_tokens=True)

        # 分行：保留标点（每行 xx，或 xx。）
        raw_lines = re.split(r"(?<=[。，！？；、])", poem)
        raw_lines = [l.strip() for l in raw_lines if l.strip()]

        # 提取纯汉字行，标定字数
        parsed = []  # [(汉字串, 标点)]
        for line in raw_lines:
            chars = re.findall(r"[一-鿿]", line)
            punct = line[-1] if line and line[-1] in "，。！？；、" else "。"
            if chars:
                parsed.append(("".join(chars), punct))

        # 筛选符合字数的行
        valid = [(ch, p) for ch, p in parsed
                 if len(ch) == expected_len or bart_style == "song"]
        if len(valid) < num_lines:
            valid = parsed[:num_lines] if len(parsed) >= num_lines else parsed

        # === 第二阶段：韵脚约束修正 ===
        if bart_style != "song" and len(valid) >= 2:
            valid = self._fix_rhyme_with_embedding(
                valid, tok, mdl, expected_len, num_lines
            )

        # 拼接输出
        result_lines = []
        for i, (chars, punct) in enumerate(valid[:num_lines]):
            result_lines.append(chars + punct)

        return "\n".join(result_lines)

    def _fix_rhyme_with_embedding(self, lines: list, tok, mdl,
                                   chars_per_line: int, num_lines: int) -> list:
        # 确保 lines 内都是 (chars, punct) 元组
        fixed = []
        for item in lines:
            if isinstance(item, tuple) and len(item) == 2:
                fixed.append(item)
            elif isinstance(item, str):
                chars = re.findall(r'[一-鿿]', item)
                punct = item[-1] if item and item[-1] in '，。！？；、' else '。'
                fixed.append((''.join(chars), punct))
        lines = fixed
        """
        用 BART embedding 相似度修正韵脚：
        1. 从第2行末字提取目标韵母
        2. 对偶数行(4,6,8)末字，若不押韵则替换为语义最接近的押韵字
        """
        from backend.utils.rhyme_dict import get_final, get_rhyming_chars

        if len(lines) < 2:
            return lines

        # 从第2行获取目标韵母
        ch2 = lines[1][0][-1] if len(lines[1][0]) > 0 else ""
        target_final = get_final(ch2)
        if not target_final:
            return lines

        rhyming_pool = get_rhyming_chars(target_final, top_n=60)
        if len(rhyming_pool) < 3:
            return lines

        # 获取 BART 的 input embedding 矩阵
        embed_weight = mdl.get_input_embeddings().weight  # [vocab_size, hidden_dim]

        def find_best_rhyme_char(original_char: str, candidates: list[str]) -> str:
            """在押韵候选字中，找 embedding 与原字最接近的"""
            # 获取原字的 embedding（取第一个 token）
            orig_ids = tok.encode(original_char, add_special_tokens=False)
            if not orig_ids:
                return original_char
            orig_emb = embed_weight[orig_ids[0]]  # [hidden_dim]

            best_char = original_char
            best_sim = -999

            for cand in candidates:
                if cand == original_char:
                    return cand  # 已经押韵
                cand_ids = tok.encode(cand, add_special_tokens=False)
                if not cand_ids:
                    continue
                cand_emb = embed_weight[cand_ids[0]]
                # 余弦相似度
                sim = torch.cosine_similarity(orig_emb, cand_emb, dim=0).item()
                if sim > best_sim:
                    best_sim = sim
                    best_char = cand

            return best_char

        # 修正偶数行（index 1,3,5,7）
        for i in range(1, min(len(lines), num_lines), 2):
            chars, punct = lines[i]
            if len(chars) != chars_per_line:
                continue

            last_char = chars[-1]
            cur_final = get_final(last_char)

            # 检查是否已经押韵
            already_rhymes = False
            from backend.utils.rhyme_dict import _is_rhyme_match
            if _is_rhyme_match(cur_final, target_final):
                already_rhymes = True

            if already_rhymes:
                continue

            # 找最佳替换字
            new_char = find_best_rhyme_char(last_char, rhyming_pool)
            if new_char != last_char:
                new_chars = chars[:-1] + new_char
                lines[i] = (new_chars, punct)

        return lines

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

    def generate(self, prompt: str, num_lines: int = 4, temperature: float = 0.8,
                 style: str = "tang", num_candidates: int = 4) -> str:
        if self.model is None:
            raise RuntimeError("模型未加载，请先调用 load_model()")

        if self.model_type == "bart":
            # 映射 style
            bart_style = style if style in self._bart_models else "wuyan"
            return self._generate_bart(prompt, num_lines, temperature, bart_style)
        elif self.model_type == "gpt2":
            # 自由生成候选，选最优
            candidates = []
            for i in range(num_candidates):
                temp = temperature + i * 0.1
                candidates.append(
                    self._generate_gpt2(prompt, num_lines, temp, style="free"))
            return self._pick_best(candidates, prompt)
        else:
            result = self._generate_lstm(prompt, num_lines, temperature)
            if result is None:
                return self._fallback_generate(prompt, num_lines)
            return result

    def _pick_best(self, candidates: list[str], prompt: str) -> str:
        """从多个候选诗中选择最优：押韵 + 格式规整 + 字数达标"""
        from backend.utils.rhyme_checker import RhymeChecker
        import re

        rhyme = RhymeChecker()
        best = candidates[0]
        best_score = -1

        for cand in candidates:
            # 按标点分行
            parts = []
            buf = ''
            for ch in cand:
                buf += ch
                if ch in '，。！？':
                    if buf.strip(): parts.append(buf.strip()); buf = ''

            if len(parts) < 2:
                continue

            # 每句汉字数
            char_lens = [len(re.findall(r'[一-鿿]', l)) for l in parts]
            target = max(set(char_lens), key=char_lens.count) if char_lens else 5
            format_ok = all(l == target for l in char_lens) and target in (5, 7)

            # 押韵分
            rhyme_score = rhyme.score(parts) if len(parts) >= 2 else 0

            # 综合分
            score = rhyme_score * 1.0
            if format_ok:
                score += 1.0  # 格式正确奖励
            if len(parts) >= 4:
                score += 0.5  # 行数达标奖励

            if score > best_score:
                best_score = score
                best = cand

        return best

    def _generate_gpt2(self, prompt: str, num_lines: int, temperature: float,
                       style: str = "tang") -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_len = inputs["input_ids"].shape[1]
        max_length = prompt_len + num_lines * 12

        # 唐诗: 用格式约束解码（五言+七言各生成一次，选最优）
        processors = None
        if style in ("wuyan", "qiyan"):
            chars_per_line = 5 if style == "wuyan" else 7
            processors = [FormatConstrainedLogitsProcessor(
                chars_per_line, num_lines, self.tokenizer,
                self.tokenizer.pad_token_id or 0,
                self.tokenizer.sep_token_id or self.tokenizer.eos_token_id or 0,
                prompt_len,
            )]

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
                logits_processor=processors,
            )

        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # 去掉 prompt 部分
        prompt_text = self.tokenizer.decode(inputs["input_ids"][0],
                                            skip_special_tokens=True)
        if generated.startswith(prompt_text):
            generated = generated[len(prompt_text):]
        return generated.strip()

    def _generate_lstm(self, prompt: str, num_lines: int, temperature: float,
                       chars_per_line: int = 5) -> str | None:
        if not self._lstm_trained:
            return None
        if not hasattr(self.model, "word2ix") or not self.model.ix2word:
            return None
        if not self.model.word2ix:
            return None
        try:
            return self.model.generate(
                start_text=prompt,
                chars_per_line=chars_per_line,
                num_lines=num_lines,
                temperature=temperature,
                device=self.device,
            )
        except Exception as e:
            print(f"[WARN] LSTM generate error: {e}")
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
