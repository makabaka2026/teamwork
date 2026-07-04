"""数据预处理 — Seq2Seq 格式 (关键词→诗句) + 数据清洗"""
import json, re, sys, random
from pathlib import Path
from collections import Counter

try:
    from zhconv import convert
except ImportError:
    import subprocess; subprocess.check_call([sys.executable, "-m", "pip", "install", "zhconv", "-q"])
    from zhconv import convert

IMAGERY = {
    "春","夏","秋","冬","风","花","雪","月","山","水","云","雨",
    "日","夜","朝","暮","晓","夕","星","江","河","湖","海",
    "柳","梅","竹","菊","兰","松","荷","桃","李",
    "燕","雁","莺","鹤","鸦","蝉","蝶","鹃",
    "楼","亭","台","阁","桥","寺","门","窗",
    "酒","茶","琴","棋","书","画","剑","笛","箫",
    "离别","相思","思乡","怀古","送别","田园","边塞","征战",
    "长安","洛阳","江南","关山","阳关",
    "明月","清风","白云","流水","落花","飞雪",
    "登高","望远","行路","归乡","春望","秋思",
}

def to_simplified(text):
    return convert(text, "zh-cn")


def clean_poem_text(poem):
    """清洗诗句文本: 去掉残缺补字符号、缺字占位符、注释"""
    cleaned = []
    for line in poem["paragraphs"]:
        line = re.sub(r"\[([^\]]*)\]", r"\1", line)       # [X] 补字括号
        line = line.replace("□", "")                        # 缺字占位符
        line = re.sub(r"[（(]一[作云].*?[）)]", "", line)   # 古籍异文注释
        # 古籍引用注释: （见《XX》卷X） / （《XX》XX）
        line = re.sub(r"[（(]见《[^》]+》[^）)]*[）)]", "", line)
        line = re.sub(r"[（(]《[^》]+》[^）)]*[）)]", "", line)
        # 去掉全角/半角括号及其中内容 (最后兜底)
        line = re.sub(r"[（(][^）)]*[）)]", "", line)
        chars = re.findall(r"[一-鿿]", line)
        if len(chars) >= 3:
            cleaned.append(line)
    return cleaned


def extract_keywords(poem):
    """
    提取关键词，但排除出现在诗句开头前4个字的词。
    避免模型学到 "关键词 = 首句开头" 的错误映射。
    """
    text = "".join(poem["paragraphs"])
    title = poem.get("title", "")
    full = title + text

    # 诗句开头前4个汉字
    first_chars = "".join(re.findall(r"[一-鿿]", text)[:4])

    found = []
    for k in IMAGERY:
        if k in full:
            # 关键词是否出现在诗句开头？
            starts_at_beginning = (k in first_chars)
            # 优先选择: 标题中的关键词 > 诗中后部的关键词 > 开头的关键词
            if k in title:
                priority = 0
            elif not starts_at_beginning:
                priority = 1
            else:
                priority = 2  # 最低优先级 — 关键词在开头，容易导致复制
            found.append((priority, k))

    # 按优先级排序，取前3个
    found.sort(key=lambda x: x[0])
    return [k for _, k in found[:3]]


def classify_tang_format(poem):
    lengths = []
    for p in poem["paragraphs"]:
        chars = re.findall(r"[一-鿿]", p)
        if chars: lengths.append(len(chars) / 2)
    if not lengths: return "gutishi"
    cnt_5 = sum(1 for h in lengths if 4.5 <= h <= 5.5)
    cnt_7 = sum(1 for h in lengths if 6.5 <= h <= 7.5)
    total = len(lengths)
    if cnt_5 / total >= 0.8: return "wuyan"
    if cnt_7 / total >= 0.8: return "qiyan"
    return "gutishi"


def load_poems(data_dir, folder, file_pattern, poem_type):
    poems = []
    d = Path(data_dir) / folder
    if not d.exists(): return poems
    for f in sorted(d.glob(file_pattern)):
        with open(f, encoding="utf-8") as fp:
            for item in json.load(fp):
                paras = item.get("paragraphs", [])
                if len(paras) >= 2:
                    poems.append({
                        "title": item.get("title", ""),
                        "author": item.get("author", item.get("poet", "未知")),
                        "paragraphs": paras,
                        "type": poem_type,
                        "rhythmic": item.get("rhythmic", ""),
                    })
    return poems


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/raw")
    parser.add_argument("--output_dir", default="data/processed")
    parser.add_argument("--max_pairs_per_poem", type=int, default=3)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载
    print("[1/4] 加载数据...")
    tang = load_poems(data_dir, "tang", "poet.tang.*.json", "tang")
    song = load_poems(data_dir, "song_ci", "*.json", "song")
    all_poems = tang + song
    print(f"  唐诗: {len(tang)}  宋词: {len(song)}")

    # 2. 清洗 + 分类 + 关键词
    print("[2/4] 繁简转换 + 数据清洗 + 分类...")
    trad = 0
    wuyan_poems, qiyan_poems, song_poems = [], [], []
    dirty_count = 0

    for poem in all_poems:
        # 清洗诗句 (去括号/□/注释)
        orig_count = sum(len(re.findall(r"[一-鿿]", l)) for l in poem["paragraphs"])
        poem["paragraphs"] = clean_poem_text(poem)
        new_count = sum(len(re.findall(r"[一-鿿]", l)) for l in poem["paragraphs"])
        if new_count < orig_count * 0.5:  # 清洗后太短，跳过
            dirty_count += 1
            continue

        # 繁简转换
        new_paras = []
        for line in poem["paragraphs"]:
            b, a = line, to_simplified(line)
            for x, y in zip(b, a):
                if x != y: trad += 1
            new_paras.append(a)
        poem["paragraphs"] = new_paras
        poem["keywords"] = extract_keywords(poem)

        fmt = classify_tang_format(poem) if poem["type"] == "tang" else "songci"
        poem["format"] = fmt
        if fmt == "wuyan": wuyan_poems.append(poem)
        elif fmt == "qiyan": qiyan_poems.append(poem)
        elif fmt == "songci": song_poems.append(poem)

    print(f"  繁体→简体: {trad} 字符")
    print(f"  清洗掉: {dirty_count} 首 (缺字严重)")
    print(f"  五言: {len(wuyan_poems)}  七言: {len(qiyan_poems)}  宋词: {len(song_poems)}")

    # 3. Seq2Seq 训练数据
    print("[3/4] 生成 Seq2Seq 训练对...")

    def save_seq2seq(poems, out_name):
        path = output_dir / out_name
        pairs = 0
        skipped_kw = 0
        skipped_short = 0
        with open(path, "w", encoding="utf-8") as f:
            for poem in poems:
                poem_text = "\n".join(poem["paragraphs"])
                chinese_chars = re.findall(r"[一-鿿]", poem_text)
                if len(chinese_chars) < 15:
                    skipped_short += 1
                    continue

                first_n_chars = "".join(chinese_chars[:6])

                # === 模式1: 标题 → 全诗 (标题天然概括内容) ===
                fmt = poem.get("format", "")
                tag = "五言：" if "wuyan" in fmt else ("七言：" if "qiyan" in fmt else "")

                # 优先用诗题 (如 "春望", "登高", "送别")
                title = poem.get("title", "").strip()
                # 过滤无意义标题
                bad_titles = {"句","杂诗","无题","","续","句赠","杂言","杂曲","乐府","古诗"}
                if title and title not in bad_titles and len(title) >= 2:
                    clean_title = re.sub(r"[（(].*?[）)]", "", title)  # 去括号注释
                    clean_title = re.sub(r"[〈《].*?[〉》]", "", clean_title)
                    clean_title = clean_title.strip()
                    if len(clean_title) >= 2:
                        f.write(json.dumps({"input": tag + clean_title, "output": poem_text},
                                           ensure_ascii=False) + "\n")
                        pairs += 1

                # 补充关键词 (位置不在开头的)
                kws = poem.get("keywords", [])
                used_kws = [k for k in kws if k not in first_n_chars]
                if not used_kws:
                    used_kws = ["古诗"]
                for kw in used_kws[:1]:  # 最多1个关键词补充
                    f.write(json.dumps({"input": tag + kw, "output": poem_text},
                                       ensure_ascii=False) + "\n")
                    pairs += 1

                # === 模式2: 首联 → 全诗 (完整场景设定) ===
                # 提取第一联 (两句), 作为强主题锚点
                segments = re.split(r"[，。！？、；：\n]+", poem_text)
                segments = [s for s in segments if len(re.findall(r"[一-鿿]", s)) >= 3]
                if len(segments) >= 2:
                    couplet = segments[0] + "，" + segments[1] + "。"
                    fmt = poem.get("format", "")
                    tag = "五言：" if "wuyan" in fmt else ("七言：" if "qiyan" in fmt else "")
                    # 首联 → 全诗 (教会模型从场景出发保持主题连贯)
                    f.write(json.dumps({"input": tag + couplet, "output": poem_text},
                                       ensure_ascii=False) + "\n")
                    pairs += 1

        print(f"  {out_name}: {pairs} 对 ({len(poems)}首, "
              f"跳过开头词:{skipped_kw}, 跳过短诗:{skipped_short})")

    save_seq2seq(wuyan_poems, "seq2seq_wuyan.jsonl")
    save_seq2seq(qiyan_poems, "seq2seq_qiyan.jsonl")
    save_seq2seq(wuyan_poems + qiyan_poems, "seq2seq_tang.jsonl")
    save_seq2seq(song_poems, "seq2seq_songci.jsonl")

    # 4. 验证数据质量
    print("[4/4] 数据验证:")
    for fname in ["seq2seq_wuyan.jsonl","seq2seq_qiyan.jsonl","seq2seq_tang.jsonl"]:
        fp = output_dir / fname
        if not fp.exists(): continue
        with open(fp) as f:
            lines = sum(1 for _ in f)
        size_mb = fp.stat().st_size / 1024 / 1024

        # 抽查污染
        with open(fp) as f:
            dirty = sum(1 for l in f if "[" in l or "□" in l or "（" in l)
        print(f"  {fname}: {lines} 条, {size_mb:.1f}MB, 残留污染: {dirty}")

    print("\n预处理完成。训练命令: python3 train_bart.py --data_file seq2seq_tang.jsonl")


if __name__ == "__main__":
    main()
