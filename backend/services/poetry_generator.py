"""诗词生成核心 — GPT-2 / LSTM"""
import torch
from backend.config import Config


class PoetryGenerator:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_type = "none"
        self.gpu_available = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.gpu_available else "cpu")

    def load_model(self, model_type: str = "gpt2"):
        """加载模型，优先 GPT-2，失败则回退 LSTM"""
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
        """加载 GPT2-Chinese-Poem 模型"""
        from transformers import GPT2LMHeadModel, BertTokenizer

        model_name = "uer/gpt2-chinese-poem"
        print(f"[INFO] 加载 GPT-2 模型: {model_name} ...")

        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        print(f"[INFO] GPT-2 加载完成 (device={self.device})")

    def _load_lstm(self):
        """加载或初始化 LSTM 模型"""
        from backend.services.lstm_model import PoemLSTM

        print("[INFO] 初始化 LSTM 模型 ...")
        self.model = PoemLSTM(
            vocab_size=8000,
            embedding_dim=256,
            hidden_dim=512,
            num_layers=3,
        )
        self.model.to(self.device)
        self.model.eval()

        # 尝试加载预训练权重
        ckpt_path = Config.LSTM_MODEL_PATH
        if ckpt_path.exists():
            print(f"[INFO] 加载 LSTM 权重: {ckpt_path}")
            self.model.load_state_dict(
                torch.load(ckpt_path, map_location=self.device, weights_only=True)
            )
        else:
            print("[WARN] 未找到 LSTM 权重，使用随机初始化（请先训练）")

        self.tokenizer = None  # LSTM 使用 char-level vocab
        print(f"[INFO] LSTM 加载完成 (device={self.device})")

    def switch_model(self, model_type: str):
        """运行时切换模型"""
        if self.model_type != model_type:
            self.load_model(model_type)

    def generate(
        self, prompt: str, num_lines: int = 4, temperature: float = 0.8
    ) -> str:
        """生成诗词文本"""
        if self.model is None:
            raise RuntimeError("模型未加载，请先调用 load_model()")

        if self.model_type == "gpt2":
            return self._generate_gpt2(prompt, num_lines, temperature)
        else:
            return self._generate_lstm(prompt, num_lines, temperature)

    def _generate_gpt2(self, prompt: str, num_lines: int, temperature: float) -> str:
        max_len_per_line = 10  # 每句约 5-7 字 + 标点
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
        # 去掉 prompt 部分
        if generated.startswith(prompt):
            generated = generated[len(prompt):]
        return generated.strip()

    def _generate_lstm(self, prompt: str, num_lines: int, temperature: float) -> str:
        # LSTM 热启动模式：用 prompt 中的关键字作为起始字符
        # 这里使用简化实现
        if not hasattr(self.model, "word2ix") or not hasattr(self.model, "ix2word"):
            return _FALLBACK_POEMS.get(prompt[:2], "春眠不觉晓，\n处处闻啼鸟。\n夜来风雨声，\n花落知多少。")
        return self.model.generate(
            start_text=prompt,
            max_chars=num_lines * 10,
            temperature=temperature,
            device=self.device,
        )


# 回退诗词库（模型未就绪时使用）
_FALLBACK_POEMS = {
    "春": "春风得意马蹄疾，\n一日看尽长安花。\n等闲识得东风面，\n万紫千红总是春。",
    "月": "床前明月光，\n疑是地上霜。\n举头望明月，\n低头思故乡。",
    "花": "花开堪折直须折，\n莫待无花空折枝。\n落红不是无情物，\n化作春泥更护花。",
    "山": "会当凌绝顶，\n一览众山小。\n横看成岭侧成峰，\n远近高低各不同。",
    "水": "飞流直下三千尺，\n疑是银河落九天。\n问渠那得清如许，\n为有源头活水来。",
}
