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
        max_chars: int = 40,
        temperature: float = 0.8,
        device: torch.device = None,
    ) -> str:
        """从起始文本逐字生成诗词"""
        if device is None:
            device = next(self.parameters()).device

        self.eval()

        # 将起始文本转为索引
        indices = [self.word2ix.get(ch, 0) for ch in start_text]
        if not indices:
            indices = [0]

        input_tensor = torch.tensor([indices], device=device)
        hidden = self.init_hidden(1, device)

        generated = list(start_text)

        with torch.no_grad():
            for _ in range(max_chars - len(start_text)):
                logits, hidden = self.forward(input_tensor, hidden)
                logits = logits[:, -1, :] / max(temperature, 0.01)

                probs = torch.softmax(logits, dim=-1)
                next_idx = torch.multinomial(probs, 1).item()

                ch = self.ix2word.get(next_idx, "")
                if ch in ["", "<EOS>", "<PAD>"]:
                    break
                generated.append(ch)
                input_tensor = torch.tensor([[next_idx]], device=device)

        return "".join(generated)
