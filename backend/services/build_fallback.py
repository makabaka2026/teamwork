"""从御定全唐诗数据构建回退诗库 + LSTM词汇表"""
import json
import pickle
import re
from pathlib import Path
from collections import defaultdict, Counter

RAW_DIR = Path(__file__).resolve().parent.parent.parent / "training" / "data" / "raw" / "tang"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 解析所有唐诗 ──
all_poems = []  # [(title, author, lines)]
char_counter = Counter()

for json_file in sorted(RAW_DIR.glob("*.json")):
    with open(json_file, encoding="utf-8") as f:
        data = json.load(f)
    for item in data:
        paragraphs = item.get("paragraphs", [])
        if not paragraphs or len(paragraphs) < 2:
            continue
        # 清理每行（去标点以外的空格等）
        lines = [line.strip() for line in paragraphs if line.strip()]
        if len(lines) < 2:
            continue
        title = item.get("title", "").strip()
        author = item.get("author", "").strip()
        all_poems.append((title, author, lines))
        for line in lines:
            for ch in line:
                if "一" <= ch <= "鿿":
                    char_counter[ch] += 1

print(f"共解析 {len(all_poems)} 首唐诗")
print(f"共 {len(char_counter)} 个不重复汉字")

# ── 构建关键词索引 ──
# 从每首诗第一句提取关键词（第一个实义词），按词索引
keyword_index = defaultdict(list)  # keyword -> [poem_lines, ...]

# 预定义高频主题关键词
THEME_KEYWORDS = [
    "春", "夏", "秋", "冬", "月", "花", "山", "水", "风", "云",
    "雪", "雨", "日", "夜", "江", "海", "天", "人", "心", "梦",
    "酒", "歌", "剑", "马", "柳", "竹", "松", "梅", "兰", "菊",
    "桃", "荷", "鹤", "雁", "燕", "莺", "蝉", "鱼", "龙", "凤",
    "边塞", "故乡", "离别", "相思", "长安", "洛阳", "江南",
]

for title, author, lines in all_poems:
    first_line = lines[0]
    # 按主题关键词索引
    for kw in THEME_KEYWORDS:
        if kw in first_line or kw in title:
            keyword_index[kw].append(lines)
            break
    # 用首句第一个汉字索引
    for ch in first_line:
        if "一" <= ch <= "鿿":
            keyword_index[ch].append(lines)
            break

# 只保留有足够诗词的关键词（至少3首）
keyword_index = {k: v for k, v in keyword_index.items() if len(v) >= 3}
print(f"共 {len(keyword_index)} 个关键词索引，覆盖 {sum(len(v) for v in keyword_index.values())} 首")

# ── 保存回退诗库 ──
fallback_path = OUTPUT_DIR / "fallback_poems.json"
fallback_data = {k: v[:20] for k, v in keyword_index.items()}  # 每个关键词最多20首
with open(fallback_path, "w", encoding="utf-8") as f:
    json.dump(fallback_data, f, ensure_ascii=False, indent=2)
print(f"回退诗库已保存: {fallback_path} ({len(fallback_data)} 个关键词)")

# ── 构建LSTM词汇表 ──
VOCAB_SIZE = 8000
most_common = char_counter.most_common(VOCAB_SIZE - 4)

word2ix = {"<PAD>": 0, "<START>": 1, "<EOS>": 2, "<UNK>": 3}
for i, (ch, _) in enumerate(most_common, start=4):
    word2ix[ch] = i
ix2word = {v: k for k, v in word2ix.items()}

vocab_path = OUTPUT_DIR / "vocab.pkl"
with open(vocab_path, "wb") as f:
    pickle.dump({"word2ix": word2ix, "ix2word": ix2word}, f)
print(f"词汇表已保存: {vocab_path} ({len(word2ix)} 个字符)")

# ── 统计 ──
print(f"\n=== 完成 ===")
print(f"唐诗总数: {len(all_poems)} 首")
print(f"回退诗库关键词: {len(fallback_data)} 个")
print(f"词汇表大小: {len(word2ix)} 个字符")
print(f"\n文件位置:")
print(f"  回退诗库: {fallback_path}")
print(f"  LSTM词汇: {vocab_path}")
