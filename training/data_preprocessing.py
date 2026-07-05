"""数据预处理 — 关键词+场景描述 → 诗句 (Seq2Seq) + 数据清洗"""
import json, re, sys, random
from pathlib import Path
from collections import Counter

try:
    from zhconv import convert
except ImportError:
    import subprocess; subprocess.check_call([sys.executable, "-m", "pip", "install", "zhconv", "-q"])
    from zhconv import convert

# 无效标题 (太泛，无法描述内容)
BAD_TITLES = {"句","杂诗","无题","","续","句赠","杂言","杂曲","乐府","古诗","咏","感怀","杂兴","偶题","即事","有感","口号","漫兴","书怀","遣怀"}

# 诗题→场景描述的映射模板
SCENE_TEMPLATES = {
    "登高": "登高远望，天地辽阔",
    "春望": "春天登高眺望",
    "送别": "长亭送别，依依不舍",
    "相思": "深深的思念之情",
    "离别": "离别时的愁绪",
    "边塞": "边塞风光，征战沙场",
    "闺怨": "女子独守空闺的哀怨",
    "思乡": "思念故乡的愁绪",
    "怀古": "凭吊古迹，感怀历史",
    "咏史": "咏叹历史兴亡",
    "山水": "山水田园的优美风光",
    "田园": "田园生活的闲适",
    "归乡": "归乡途中的感慨",
    "征战": "征战沙场的壮烈",
    "行军": "行军途中的艰辛",
    "春夜": "春夜的静谧与思念",
    "秋思": "秋日的感怀与思念",
    "夜雨": "夜雨中的愁思",
    "秋日": "秋日的景色与感怀",
}

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
    """清洗诗句文本: 去掉残缺补字符号、缺字占位符、所有注释"""
    cleaned = []
    for line in poem["paragraphs"]:
        # [X] 补字括号 → 保留文字
        line = re.sub(r"\[([^\]]*)\]", r"\1", line)
        line = line.replace("□", "")

        # 全角括号 — 只要括号内包含这些关键词，整段括号+内容全删
        citation_kw = r"卷|收|见|全唐|永乐|诗|录|集|典|部|注|按|一作|一本|或作"
        line = re.sub(r"[（(][^）)]*(?:" + citation_kw + r")[^）)]*[）)]", "", line)

        # 半角括号
        line = re.sub(r"\([^)]*(?:" + citation_kw + r")[^)]*\)", "", line)

        # 兜底: 移除所有残留全角/半角括号内容
        line = re.sub(r"[（(][^）)]*[）)]", "", line)
        line = re.sub(r"\([^)]*\)", "", line)

        # 去掉只有标点符号的行
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


def generate_scene_description(poem):
    """
    生成任务式场景描述 — 明确是"创作指令"而非诗的开头。
    输出如: "写一首五言诗，描写离别时的愁绪和不舍"
    """
    title = poem.get("title", "").strip()
    clean_title = re.sub(r"[（(].*?[）)]", "", title).strip()
    kws = poem.get("keywords", [])[:4]
    fmt = poem.get("format", "")
    fmt_name = "五言诗" if "wuyan" in fmt else ("七言诗" if "qiyan" in fmt else ("词" if "songci" in fmt else "诗"))

    # 1. 使用预定义模板 + 格式
    for tpl_key, tpl_desc in SCENE_TEMPLATES.items():
        if tpl_key in clean_title:
            return f"写一首{fmt_name}，{tpl_desc}。"

    # 2. 诗题描述
    if clean_title and clean_title not in BAD_TITLES and len(clean_title) >= 2:
        return f"写一首{fmt_name}，主题是{clean_title}。"

    # 3. 意象描述
    if kws:
        return f"写一首{fmt_name}，意象：{'、'.join(kws[:4])}。"

    return f"写一首{fmt_name}。"


def classify_tang_format(poem):
    lengths = []
    for p in poem["paragraphs"]:
        chars = re.findall(r"[一-鿿]", p)
        if chars: lengths.append(len(chars) / 2)
    if not lengths: return "gutishi"
    cnt_5 = sum(1 for h in lengths if 4.5 <= h <= 5.5)
    cnt_7 = sum(1 for h in lengths if 6.5 <= h <= 7.5)
    total = len(lengths)
    if cnt_5 / total >= 0.95: return "wuyan"
    if cnt_7 / total >= 0.95: return "qiyan"
    # 严格: 所有行必须统一长度
    if cnt_5 + cnt_7 < total:
        return "gutishi"
    if cnt_5 > cnt_7: return "wuyan"
    return "qiyan"


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
        """保存 Seq2Seq 训练数据: 标题/关键词/首联/场景 → 全诗"""
        path = output_dir / out_name
        pairs = 0
        skipped_short = 0
        with open(path, "w", encoding="utf-8") as f:
            for poem in poems:
                poem_text = "\n".join(poem["paragraphs"])
                chinese_chars = re.findall(r"[一-鿿]", poem_text)
                if len(chinese_chars) < 15:
                    skipped_short += 1
                    continue

                fmt = poem.get("format", "")
                tag = ""  # 五言/七言分开训练, 不需要格式标签

                # === 模式1: 场景描述 → 全诗 (指令格式) ===
                scene = generate_scene_description(poem)
                # 指令格式: "写诗：xxx" — 教会模型输入≠诗句
                f.write(json.dumps({"input": f"写诗：{scene}", "output": poem_text},
                                   ensure_ascii=False) + "\n")
                pairs += 1

                # === 模式2: 标题 → 全诗 (指令格式) ===
                title = poem.get("title", "").strip()
                clean_title = re.sub(r"[（(].*?[）)]", "", title).strip()
                if clean_title and clean_title not in BAD_TITLES and len(clean_title) >= 2:
                    f.write(json.dumps({"input": f"写诗：{clean_title}",
                                        "output": poem_text}, ensure_ascii=False) + "\n")
                    pairs += 1

                # === 模式3: 首联 → 全诗 ===
                segments = re.split(r"[，。！？、；：\n]+", poem_text)
                segments = [s for s in segments if len(re.findall(r"[一-鿿]", s)) >= 3]
                if len(segments) >= 2:
                    couplet = segments[0] + "，" + segments[1] + "。"
                    f.write(json.dumps({"input": f"续诗：{couplet}",
                                        "output": poem_text}, ensure_ascii=False) + "\n")
                    pairs += 1

        print(f"  {out_name}: {pairs} 对 ({len(poems)}首, 跳过短诗:{skipped_short})")

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
