"""快速演示：用小数据集快速训练并展示效果"""
import sys, pickle, json, random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend.services.lstm_model import PoemLSTM

DATA_DIR = Path("data/raw")
DEVICE = torch.device("cpu")

# 1. 加载少量数据
def load_small():
    poems = []
    for f in list((DATA_DIR / "tang").glob("poet.tang.*.json"))[:2]:
        for item in json.load(open(f, encoding="utf-8")):
            p = item.get("paragraphs", [])
            if 2 <= len(p) <= 8:
                poems.append(p)
    return random.sample(poems, min(500, len(poems)))

# 2. 构建小词汇表
def build_small_vocab(poems):
    counter = Counter()
    for poem in poems:
        for line in poem:
            for ch in line:
                if "一" <= ch <= "鿿":
                    counter[ch] += 1
    word2ix = {"<PAD>": 0, "<START>": 1, "<EOS>": 2, "<UNK>": 3}
    for i, (ch, _) in enumerate(counter.most_common(2000), start=4):
        word2ix[ch] = i
    return word2ix, {v: k for k, v in word2ix.items()}

# 3. 转序列
def to_sequences(poems, word2ix, seq_len=64):
    seqs = []
    for poem in poems:
        chars = [word2ix["<START>"]]
        for line in poem:
            for ch in line:
                if "一" <= ch <= "鿿":
                    chars.append(word2ix.get(ch, word2ix["<UNK>"]))
        chars.append(word2ix["<EOS>"])
        if len(chars) < seq_len:
            chars += [0] * (seq_len - len(chars))
        else:
            chars = chars[:seq_len]
        seqs.append(chars)
    return np.array(seqs, dtype=np.int64)

print("加载数据...")
poems = load_small()
print(f"使用 {len(poems)} 首诗做快速演示")

word2ix, ix2word = build_small_vocab(poems)
vocab_size = len(word2ix)
print(f"词汇量: {vocab_size}")

seqs = to_sequences(poems, word2ix)
dataset = TensorDataset(torch.tensor(seqs, dtype=torch.long))
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# 4. 快速训练
model = PoemLSTM(vocab_size=vocab_size, embedding_dim=128, hidden_dim=256, num_layers=2)
model.word2ix = word2ix
model.ix2word = ix2word

opt = Adam(model.parameters(), lr=0.003)
criterion = nn.CrossEntropyLoss(ignore_index=0)

print("\n开始快速训练 (15 epochs)...")
for epoch in range(15):
    model.train()
    total_loss = 0
    for x, in loader:
        inputs, targets = x[:, :-1], x[:, 1:]
        opt.zero_grad()
        logits, _ = model(inputs)
        loss = criterion(logits.reshape(-1, vocab_size), targets.reshape(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(loader)
    if epoch % 3 == 0:
        print(f"  Epoch {epoch+1:2d}/15, Loss: {avg_loss:.4f}")

# 5. 生成演示
print("\n" + "=" * 50)
print("  生成效果演示")
print("=" * 50)

model.eval()
prompts = ["春", "月", "风", "花"]
for prompt in prompts:
    result = model.generate(prompt, max_chars=24, temperature=0.8, device=DEVICE)
    print(f"\n  输入: {prompt}")
    print(f"  生成: {result}")
    print("  " + "-" * 40)

print("\n注意: 这只是 500 首诗训练 15 轮的快速演示，效果有限。")
print("全量数据 20 epoch 训练后效果会明显更好。")
