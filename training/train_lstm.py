"""LSTM 诗词模型训练"""
import pickle
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam

# 将 backend 加入 path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.services.lstm_model import PoemLSTM


def load_data(data_dir: Path):
    """加载预处理后的数据"""
    data_path = data_dir / "tang_lstm.npz"
    vocab_path = data_dir / "vocab.pkl"

    if not data_path.exists():
        raise FileNotFoundError(f"数据文件不存在: {data_path}，请先运行 data_preprocessing.py")

    data = np.load(data_path)
    sequences = data["data"]

    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)

    return sequences, vocab["word2ix"], vocab["ix2word"]


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] 训练设备: {device}")

    # 加载数据
    sequences, word2ix, ix2word = load_data(Path(args.data_dir))
    vocab_size = len(word2ix)

    # 划分训练/验证集
    np.random.seed(42)
    np.random.shuffle(sequences)
    split = int(len(sequences) * 0.9)
    train_data = sequences[:split]
    val_data = sequences[split:]

    train_dataset = TensorDataset(torch.tensor(train_data, dtype=torch.long))
    val_dataset = TensorDataset(torch.tensor(val_data, dtype=torch.long))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    # 创建模型
    model = PoemLSTM(
        vocab_size=vocab_size,
        embedding_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
    ).to(device)

    # 注入词汇表
    model.word2ix = word2ix
    model.ix2word = ix2word

    optimizer = Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略 PAD

    # 训练循环
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0

        for batch_idx, (x,) in enumerate(train_loader):
            x = x.to(device)
            # x: (batch, seq_len), 目标: 预测下一个字符
            inputs = x[:, :-1]
            targets = x[:, 1:]

            optimizer.zero_grad()
            logits, _ = model(inputs)
            loss = criterion(
                logits.reshape(-1, vocab_size),
                targets.reshape(-1),
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 50 == 0:
                print(f"  Epoch {epoch+1}/{args.epochs}, "
                      f"Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)

        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, in val_loader:
                x = x.to(device)
                inputs, targets = x[:, :-1], x[:, 1:]
                logits, _ = model(inputs)
                loss = criterion(logits.reshape(-1, vocab_size), targets.reshape(-1))
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)

        print(f"  Epoch {epoch+1}: Train Loss={avg_loss:.4f}, Val Loss={avg_val_loss:.4f}")

    # 保存模型
    output_path = Path(args.output) / "lstm_tang.pth"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    print(f"[INFO] 模型已保存: {output_path}")

    # 保存词汇表
    vocab_output = output_path.parent / "vocab.pkl"
    with open(vocab_output, "wb") as f:
        pickle.dump({"word2ix": word2ix, "ix2word": ix2word}, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--output", default="output")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--clip", type=float, default=5.0)
    args = parser.parse_args()

    print("=" * 50)
    print("  LSTM 诗词模型训练")
    print("=" * 50)
    train(args)


if __name__ == "__main__":
    main()
