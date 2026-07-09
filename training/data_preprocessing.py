"""数据集预处理 — SongNet式位置符号 + 韵脚标记 + 繁简转换 + LSTM训练格式"""
import json
import pickle
import random
import re
from pathlib import Path
from collections import Counter

import numpy as np
import zhconv


# SongNet 位置符号: p₆ p₅ p₄ p₃ p₂ p₁ p₀
# p₀ = 句末标点位, p₁ = 韵脚位
POS_TOKENS = ["<p6>", "<p5>", "<p4>", "<p3>", "<p2>", "<p1>", "<p0>"]
RHYME_TOKENS = ["<r>", "</r>"]  # 韵脚位置标记


def load_tang_poems(data_dir: Path) -> list[tuple[int, list[str]]]:
    """加载唐诗，解析为 (每句字数, 行列表)，繁体转简体"""
    poems = []
    tang_dir = data_dir / "tang"
    if not tang_dir.exists():
        print(f"[WARN] 唐诗数据目录不存在: {tang_dir}")
        return poems

    for json_file in sorted(tang_dir.glob("*.json")):
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            for item in data:
                paras = item.get("paragraphs", [])
                if len(paras) not in (2, 4):
                    continue
                all_lines = []
                for p in paras:
                    # 繁体转简体
                    p = zhconv.convert(p, "zh-cn")
                    parts = re.split(r'([，。])', p)
                    for j in range(0, len(parts) - 1, 2):
                        line = parts[j] + (parts[j+1] if j+1 < len(parts) else '')
                        chars = re.findall(r'[一-鿿]', line)
                        if len(chars) in (5, 7):
                            all_lines.append((len(chars), line))
                if len(all_lines) not in (4, 8):
                    continue
                lens = [l for l, _ in all_lines]
                if all(l == lens[0] for l in lens):
                    poems.append((lens[0], [line for _, line in all_lines]))

    print(f"[INFO] 加载唐诗(简体,格式规整): {len(poems)} 首")
    return poems


def build_vocab(poems: list, vocab_size: int = 8000) -> dict:
    """构建词汇表（包含位置符号 + 标点）"""
    ALLOWED_PUNCT = set("，。！？；、")

    # 预留 token
    reserved = ["<PAD>", "<START>", "<EOS>", "<UNK>"]
    for p in ALLOWED_PUNCT:
        reserved.append(p)
    for pt in POS_TOKENS:
        reserved.append(pt)

    # 统计汉字
    counter = Counter()
    for _, lines in poems:
        for line in lines:
            for ch in line:
                if "一" <= ch <= "鿿":
                    counter[ch] += 1

    most_common = counter.most_common(vocab_size - len(reserved))

    word2ix = {}
    for i, token in enumerate(reserved):
        word2ix[token] = i

    idx = len(reserved)
    for ch, _ in most_common:
        if ch not in word2ix:
            word2ix[ch] = idx
            idx += 1

    ix2word = {v: k for k, v in word2ix.items()}
    print(f"[INFO] 词汇量: {len(word2ix)} (含{len(POS_TOKENS)}个位置符号)")
    return word2ix, ix2word


def poem_to_songnet_sequence(
    chars_per_line: int,
    lines: list[str],
    word2ix: dict,
    seq_len: int = 128,
) -> list[int]:
    """
    SongNet 式编码 + 韵脚标记:
    偶数行(2,4)的 p₀ 位置字用 <r>...</r> 包裹，标记押韵位置

    五言: <START> <p4>春 <p3>风 <p2>吹 <p1>不 <p0>尽 ，
           <p4>玉 <p3>门 <p2>关 <p1>不 <r><p0>住</r> 。  ← 押韵位
           <p4>何 <p3>日 <p2>是 <p1>归 <p0>年 ，
           <p4>故 <p3>园 <p2>无 <r><p1>此</r> <p0>声 。  ← 押韵位(与上句偶数行同韵)
    """
    seq = [word2ix["<START>"]]

    for line_no, line in enumerate(lines):
        chars = re.findall(r'[一-鿿]', line)
        if len(chars) != chars_per_line:
            chars = chars[:chars_per_line]

        is_even = (line_no + 1) % 2 == 0  # 偶数行=押韵行
        max_pos = chars_per_line - 1

        for i, ch in enumerate(chars):
            pos = max_pos - i

            # 偶数行的 p₀ 位置: 韵脚字，前后加 <r> 标记
            if is_even and pos == 0:
                seq.append(word2ix.get("<r>", word2ix["<UNK>"]))
                seq.append(word2ix.get(f"<p{pos}>", word2ix["<UNK>"]))
                seq.append(word2ix.get(ch, word2ix["<UNK>"]))
                seq.append(word2ix.get("</r>", word2ix["<UNK>"]))
            else:
                seq.append(word2ix.get(f"<p{pos}>", word2ix["<UNK>"]))
                seq.append(word2ix.get(ch, word2ix["<UNK>"]))

        punct = "。" if is_even else "，"
        seq.append(word2ix.get(punct, word2ix["<UNK>"]))

    seq.append(word2ix["<EOS>"])

    if len(seq) < seq_len:
        seq += [word2ix["<PAD>"]] * (seq_len - len(seq))
    else:
        seq = seq[:seq_len]

    return seq


def poems_to_sequences(
    poems: list[tuple[int, list[str]]],
    word2ix: dict,
    seq_len: int = 128,
) -> np.ndarray:
    """将所有诗词转为 SongNet 式序列"""
    sequences = []
    for chars_per_line, lines in poems:
        seq = poem_to_songnet_sequence(chars_per_line, lines, word2ix, seq_len)
        sequences.append(seq)
    return np.array(sequences, dtype=np.int64)


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

    poems = load_tang_poems(data_dir)

    if not poems:
        print("[WARN] 没有找到诗词数据！")
        return

    # 统计
    lens = Counter(c for c, _ in poems)
    print(f"[INFO] 总诗词数: {len(poems)} (五言={lens[5]}, 七言={lens[7]})")

    if args.format in ("lstm", "both"):
        word2ix, ix2word = build_vocab(poems, args.vocab_size)
        sequences = poems_to_sequences(poems, word2ix, args.seq_len)

        np.savez_compressed(
            output_dir / "tang_lstm.npz",
            data=sequences, vocab_size=args.vocab_size,
        )
        with open(output_dir / "vocab.pkl", "wb") as f:
            pickle.dump({"word2ix": word2ix, "ix2word": ix2word}, f)

        print(f"[INFO] LSTM 数据: {sequences.shape}")

    print("[INFO] 数据预处理完成")


if __name__ == "__main__":
    main()
