"""LSTM 诗词模型训练"""
import pickle
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend.services.lstm_model import PoemLSTM


def load_data(data_dir: Path, prefix: str = "tang_all"):
    """加载预处理后的 LSTM 数据"""
    data_path = data_dir / f"{prefix}_lstm.npz"
    vocab_path = data_dir / f"{prefix}_vocab.pkl"

    if not data_path.exists():
        raise FileNotFoundError(f"数据文件不存在: {data_path}，请先运行 data_preprocessing.py")

    data = np.load(data_path)
    sequences = data["data"]
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)
    return sequences, vocab["word2ix"], vocab["ix2word"]


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] 设备: {device}")

    sequences, word2ix, ix2word = load_data(Path(args.data_dir), args.data_prefix)
    vocab_size = len(word2ix)
    print(f"[INFO] 样本数: {len(sequences)}, 词汇量: {vocab_size}")

    # 划分训练/验证集
    np.random.seed(42)
    np.random.shuffle(sequences)
    split = int(len(sequences) * 0.9)
    train_data = sequences[:split]
    val_data = sequences[split:]

    train_loader = DataLoader(
        TensorDataset(torch.tensor(train_data, dtype=torch.long)),
        batch_size=args.batch_size, shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(val_data, dtype=torch.long)),
        batch_size=args.batch_size,
    )

    model = PoemLSTM(
        vocab_size=vocab_size, embedding_dim=args.embed_dim,
        hidden_dim=args.hidden_dim, num_layers=args.num_layers,
    ).to(device)
    model.word2ix = word2ix
    model.ix2word = ix2word

    optimizer = Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for batch_idx, (x,) in enumerate(train_loader):
            x = x.to(device)
            inputs, targets = x[:, :-1], x[:, 1:]
            optimizer.zero_grad()
            logits, _ = model(inputs)
            loss = criterion(logits.reshape(-1, vocab_size), targets.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            total_loss += loss.item()

            if batch_idx % 50 == 0:
                print(f"  Epoch {epoch+1}/{args.epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for (x,) in val_loader:
                x = x.to(device)
                inputs, targets = x[:, :-1], x[:, 1:]
                logits, _ = model(inputs)
                val_loss += criterion(logits.reshape(-1, vocab_size), targets.reshape(-1)).item()

        print(f"  Epoch {epoch+1}: Train Loss={total_loss/len(train_loader):.4f}, Val Loss={val_loss/len(val_loader):.4f}")

    # 保存
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    model_path = out / f"lstm_{args.data_prefix}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"[INFO] 模型已保存: {model_path}")


def main():
    parser = argparse.ArgumentParser(description="LSTM 诗词模型训练")
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--data_prefix", default="tang_all",
                        help="数据前缀: tang_all / tang_wuyan / tang_qiyan")
    parser.add_argument("--output", default="output/lstm")
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
