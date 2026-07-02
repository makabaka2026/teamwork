"""数据集预处理 — 唐诗宋词 JSON → 训练格式"""
import json
import pickle
import random
from pathlib import Path
from collections import Counter

import numpy as np


def load_tang_poems(data_dir: Path) -> list[list[str]]:
    """从 chinese-poetry 仓库加载唐诗"""
    poems = []
    tang_dir = data_dir / "tang"

    if not tang_dir.exists():
        print(f"[WARN] 唐诗数据目录不存在: {tang_dir}")
        print("  请从 https://github.com/chinese-poetry/chinese-poetry 下载")
        print("  将 json/tang/ 目录放入 training/data/raw/tang/")
        return poems

    for json_file in tang_dir.glob("poet.tang.*.json"):
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            for item in data:
                paragraphs = item.get("paragraphs", [])
                if 2 <= len(paragraphs) <= 8:
                    poems.append(paragraphs)

    print(f"[INFO] 加载唐诗: {len(poems)} 首")
    return poems


def load_song_ci(data_dir: Path) -> list[list[str]]:
    """加载宋词"""
    poems = []
    ci_dir = data_dir / "song_ci"

    if not ci_dir.exists():
        return poems

    for json_file in ci_dir.glob("ci.song.*.json"):
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            for item in data:
                paragraphs = item.get("paragraphs", [])
                if 2 <= len(paragraphs) <= 12:
                    poems.append(paragraphs)

    print(f"[INFO] 加载宋词: {len(poems)} 首")
    return poems


def build_vocab(poems: list[list[str]], vocab_size: int = 8000) -> dict:
    """构建字符级词汇表"""
    counter = Counter()
    for poem in poems:
        for line in poem:
            for ch in line:
                if "一" <= ch <= "鿿":  # 只保留汉字
                    counter[ch] += 1

    # 取前 vocab_size-4 个字符（保留特殊 token）
    most_common = counter.most_common(vocab_size - 4)

    word2ix = {
        "<PAD>": 0,
        "<START>": 1,
        "<EOS>": 2,
        "<UNK>": 3,
    }
    for i, (ch, _) in enumerate(most_common, start=4):
        word2ix[ch] = i

    ix2word = {v: k for k, v in word2ix.items()}
    return word2ix, ix2word


def poems_to_sequences(
    poems: list[list[str]],
    word2ix: dict,
    seq_len: int = 128,
) -> np.ndarray:
    """将诗词转为固定长度序列"""
    sequences = []
    for poem in poems:
        chars = []
        for line in poem:
            chars.append(word2ix.get("<START>", 1))
            for ch in line:
                if "一" <= ch <= "鿿":
                    chars.append(word2ix.get(ch, word2ix["<UNK>"]))
            chars.append(word2ix.get("<EOS>", 2))

        # 填充或截断
        if len(chars) < seq_len:
            chars += [word2ix["<PAD>"]] * (seq_len - len(chars))
        else:
            chars = chars[:seq_len]

        sequences.append(chars)

    return np.array(sequences, dtype=np.int64)


def poems_to_gpt2_format(poems: list[list[str]], output_path: Path):
    """转换为 GPT-2 训练格式（JSON Lines）"""
    with open(output_path, "w", encoding="utf-8") as f:
        for poem in poems:
            text = "\n".join(poem)
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")

    print(f"[INFO] GPT-2 训练数据已保存: {output_path} ({len(poems)} 条)")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/raw")
    parser.add_argument("--output_dir", default="data/processed")
    parser.add_argument("--vocab_size", type=int, default=8000)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--format", choices=["lstm", "gpt2", "both"], default="both")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载数据
    tang = load_tang_poems(data_dir)
    song = load_song_ci(data_dir)
    all_poems = tang + song

    if not all_poems:
        print("[WARN] 没有找到诗词数据！请先下载数据集")
        return

    print(f"[INFO] 总诗词数: {len(all_poems)}")

    # LSTM 格式
    if args.format in ("lstm", "both"):
        print("[INFO] 构建 LSTM 词汇表...")
        word2ix, ix2word = build_vocab(all_poems, args.vocab_size)
        sequences = poems_to_sequences(all_poems, word2ix, args.seq_len)

        # 保存
        np.savez_compressed(
            output_dir / "tang_lstm.npz",
            data=sequences, vocab_size=args.vocab_size,
        )
        with open(output_dir / "vocab.pkl", "wb") as f:
            pickle.dump({"word2ix": word2ix, "ix2word": ix2word}, f)

        print(f"[INFO] LSTM 数据: {sequences.shape}, 词汇量: {len(word2ix)}")

    # GPT-2 格式
    if args.format in ("gpt2", "both"):
        poems_to_gpt2_format(all_poems, output_dir / "tang_gpt2.jsonl")

    print("[INFO] ✅ 数据预处理完成")


if __name__ == "__main__":
    main()
