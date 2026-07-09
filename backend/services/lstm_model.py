"""LSTM 诗词生成模型（满足项目 LSTM 技术要求）"""
import torch
import torch.nn as nn


class PoemLSTM(nn.Module):
    def __init__(
        self,
        vocab_size: int = 8000,
        embedding_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout,
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

        # 词汇表（训练时填充）
        self.word2ix = {}
        self.ix2word = {}

    def forward(self, x, hidden=None):
        emb = self.dropout(self.embedding(x))
        out, hidden = self.lstm(emb, hidden)
        out = self.dropout(out)
        logits = self.fc(out)
        return logits, hidden

    def init_hidden(self, batch_size: int, device: torch.device):
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        return (h0, c0)

    def generate(
        self,
        start_text: str,
        chars_per_line: int = 5,
        num_lines: int = 4,
        temperature: float = 0.8,
        device: torch.device = None,
    ) -> str:
        """
        SongNet + 韵脚记忆生成
        - 位置符号强喂: <p4><p3>...<p0>
        - 偶数行p₀前加<r>，取出韵脚字后存入rhyme_final
        - 后续<r>位置：boost同韵母的字
        """
        if device is None:
            device = next(self.parameters()).device
        self.eval()

        # 构建韵母索引(懒加载)
        if not hasattr(self, '_rhyme_index'):
            self._build_rhyme_index()

        max_pos = chars_per_line - 1
        rhyme_final = None  # 存储韵脚字的韵母

        # 起始
        prompt = list(start_text)
        indices = [self.word2ix.get("<START>", 1)]
        for ch in prompt:
            indices.append(self.word2ix.get(ch, self.word2ix.get("<UNK>", 3)))
        input_tensor = torch.tensor([indices], device=device)
        hidden = self.init_hidden(1, device)
        with torch.no_grad():
            _, hidden = self.forward(input_tensor, hidden)

        generated = []

        for line_no in range(num_lines):
            is_even = (line_no + 1) % 2 == 0

            if line_no == 0:
                first_word = list(start_text)
                start_i = 1
                generated.extend(first_word)
                for ch in first_word:
                    ch_id = self.word2ix.get(ch, 3)
                    ch_tensor = torch.tensor([[ch_id]], device=device)
                    with torch.no_grad():
                        _, hidden = self.forward(ch_tensor, hidden)
            else:
                start_i = 0

            for i in range(start_i, chars_per_line):
                pos = max_pos - i

                # 偶数行 p₀ 位置: 韵脚位
                if is_even and pos == 0:
                    # 如果有韵脚记忆，boost 同韵母的字
                    r_id = self.word2ix.get("<r>", 3)
                    r_tensor = torch.tensor([[r_id]], device=device)
                    with torch.no_grad():
                        _, hidden = self.forward(r_tensor, hidden)

                # 位置符号
                pos_token = f"<p{pos}>"
                pos_id = self.word2ix.get(pos_token, 3)
                pos_tensor = torch.tensor([[pos_id]], device=device)
                with torch.no_grad():
                    logits, hidden = self.forward(pos_tensor, hidden)
                    logits = logits[:, -1, :] / max(temperature, 0.01)

                # 压制特殊token
                for tok_id in [0, 1, 2, 3]:
                    logits[0, tok_id] = -1e9
                for pt in ["<p0>", "<p1>", "<p2>", "<p3>", "<p4>", "<p5>", "<p6>", "<r>", "</r>"]:
                    tid = self.word2ix.get(pt)
                    if tid is not None:
                        logits[0, tid] = -1e9
                for p in "，。！？；、":
                    tid = self.word2ix.get(p)
                    if tid is not None:
                        logits[0, tid] = -1e9

                # 韵脚boost: 提高同韵母字的概率
                if is_even and pos == 0 and rhyme_final is not None:
                    for ch_idx in self._rhyme_index.get(rhyme_final, []):
                        logits[0, ch_idx] += 5.0

                probs = torch.softmax(logits, dim=-1)
                next_idx = torch.multinomial(probs, 1).item()
                ch = self.ix2word.get(next_idx, "")
                if ch in ["", "<EOS>", "<PAD>"]:
                    break
                generated.append(ch)
                input_tensor = torch.tensor([[next_idx]], device=device)
                with torch.no_grad():
                    _, hidden = self.forward(input_tensor, hidden)

                # 偶数行p₀: 记住韵母
                if is_even and pos == 0:
                    rhyme_final = self._get_final(ch)
                    # 关闭韵脚标记
                    rr_id = self.word2ix.get("</r>", 3)
                    rr_tensor = torch.tensor([[rr_id]], device=device)
                    with torch.no_grad():
                        _, hidden = self.forward(rr_tensor, hidden)

            # 标点
            punct = "。" if is_even else "，"
            punct_id = self.word2ix.get(punct, 3)
            punct_tensor = torch.tensor([[punct_id]], device=device)
            with torch.no_grad():
                _, hidden = self.forward(punct_tensor, hidden)
            generated.append(punct)

        return "".join(generated)

    def _build_rhyme_index(self):
        """构建韵母 → 字符索引表"""
        from pypinyin import pinyin, Style
        self._rhyme_index = {}
        for ch, idx in self.word2ix.items():
            if len(ch) == 1 and '一' <= ch <= '鿿':
                try:
                    py = pinyin(ch, style=Style.FINALS, strict=False)
                    final = py[0][0] if py and py[0] else ''
                    if final:
                        if final not in self._rhyme_index:
                            self._rhyme_index[final] = []
                        self._rhyme_index[final].append(idx)
                except:
                    pass

    def _get_final(self, ch: str) -> str:
        from pypinyin import pinyin, Style
        try:
            py = pinyin(ch, style=Style.FINALS, strict=False)
            return py[0][0] if py and py[0] else ''
        except:
            return ''
