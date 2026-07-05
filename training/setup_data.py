"""
数据集初始化 — 解压、重命名、组织目录结构
用法: python3 setup_data.py [--dataset_dir 解压目录] [--output_dir 输出目录]
"""
import os
import re
import json
import shutil
from pathlib import Path
from collections import Counter


def decode_java_unicode(name: str) -> str:
    """将 #UXXXX 格式的 Java Unicode 转义解码为中文"""
    return re.sub(r"#U([0-9a-fA-F]{4})", lambda m: chr(int(m.group(1), 16)), name)


def rename_dirs(root: Path):
    """递归重命名所有 #UXXXX 格式的目录和文件"""
    for path in sorted(root.rglob("*"), reverse=True):
        new_name = decode_java_unicode(path.name)
        if new_name != path.name:
            new_path = path.with_name(new_name)
            path.rename(new_path)


def setup_raw_data(dataset_root: Path, output_dir: Path):
    """
    从 chinese-poetry-master 提取唐诗和宋词到 data/raw 目录
    """
    tang_dir = dataset_root / "全唐诗"
    song_dir = dataset_root / "宋词"

    # 确保目录存在
    raw_tang = output_dir / "raw" / "tang"
    raw_song = output_dir / "raw" / "song_ci"
    raw_tang.mkdir(parents=True, exist_ok=True)
    raw_song.mkdir(parents=True, exist_ok=True)

    # === 唐诗 ===
    tang_count = 0
    if tang_dir.exists():
        # 全唐诗中的 JSON 文件在 error/ 子目录下
        tang_error = tang_dir / "error"
        tang_source = tang_error if tang_error.exists() else tang_dir
        for f in tang_source.glob("poet.tang.*.json"):
            dest = raw_tang / f.name
            if not dest.exists():
                shutil.copy2(f, dest)
            tang_count += 1
    print(f"[INFO] 唐诗 JSON 文件: {tang_count}")

    # === 宋词 ===
    song_count = 0
    if song_dir.exists():
        for f in song_dir.glob("ci.song.*.json"):
            dest = raw_song / f.name
            if not dest.exists():
                shutil.copy2(f, dest)
            song_count += 1
        # 也可能有 poet.song.*.json 文件
        for f in song_dir.glob("poet.song.*.json"):
            dest = raw_song / f.name
            if not dest.exists():
                shutil.copy2(f, dest)
            song_count += 1
    print(f"[INFO] 宋词 JSON 文件: {song_count}")

    # === 统计 ===
    # 统计唐诗
    tang_poems = 0
    for f in raw_tang.glob("*.json"):
        with open(f, encoding="utf-8") as fp:
            tang_poems += len(json.load(fp))
    print(f"[INFO] 唐诗总数: {tang_poems} 首")

    # 统计宋词
    song_poems = 0
    for f in raw_song.glob("*.json"):
        with open(f, encoding="utf-8") as fp:
            song_poems += len(json.load(fp))
    print(f"[INFO] 宋词总数: {song_poems} 首")


def analyze_formats(data_dir: Path):
    """分析诗词格式分布 — 为后续分类提供数据支持"""
    from collections import Counter
    format_counter = Counter()

    # 分析唐诗
    tang_dir = data_dir / "raw" / "tang"
    if tang_dir.exists():
        for f in sorted(tang_dir.glob("*.json"))[:5]:  # 采样分析
            with open(f, encoding="utf-8") as fp:
                for item in json.load(fp):
                    paras = item.get("paragraphs", [])
                    lengths = []
                    for p in paras:
                        chars = re.findall(r"[一-鿿]", p)
                        if chars:
                            lengths.append(len(chars))
                    if lengths:
                        avg = sum(lengths) / len(lengths)
                        if all(l == 5 for l in lengths):
                            format_counter["wuyan_lvshi"] += 1  # 五言律诗
                        elif all(l == 7 for l in lengths):
                            format_counter["qiyan_lvshi"] += 1  # 七言律诗
                        elif set(lengths) == {5} and len(lengths) == 4:
                            format_counter["wuyan_jueju"] += 1  # 五言绝句
                        elif set(lengths) == {7} and len(lengths) == 4:
                            format_counter["qiyan_jueju"] += 1  # 七言绝句
                        elif len(lengths) == 4 and all(l in (5, 7) for l in lengths):
                            format_counter["mixed_jueju"] += 1
                        elif all(l in (5, 7) for l in lengths):
                            format_counter["mixed_length"] += 1
                        else:
                            format_counter["gutishi"] += 1  # 古体诗/杂言

    print("\n[INFO] 唐诗格式分布 (采样):")
    total = sum(format_counter.values()) or 1
    for fmt, count in format_counter.most_common():
        print(f"  {fmt}: {count} ({100*count/total:.1f}%)")

    return format_counter


def main():
    import argparse
    parser = argparse.ArgumentParser(description="数据集初始化")
    parser.add_argument("--dataset_dir", default="/root/autodl-tmp/chinese-poetry/chinese-poetry-master")
    parser.add_argument("--output_dir", default="data")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)

    if not dataset_root.exists():
        print(f"[ERROR] 数据集目录不存在: {dataset_root}")
        return

    print("=" * 60)
    print("步骤1: 解码文件名 (Unicode 转义 → 中文)")
    print("=" * 60)
    rename_dirs(dataset_root)
    print("[INFO] 重命名完成")

    print("\n" + "=" * 60)
    print("步骤2: 组织原始数据 (唐诗 → raw/tang, 宋词 → raw/song_ci)")
    print("=" * 60)
    setup_raw_data(dataset_root, output_dir)
    print("[INFO] 数据组织完成")

    print("\n" + "=" * 60)
    print("步骤3: 分析格式分布")
    print("=" * 60)
    analyze_formats(output_dir)

    print("\n" + "=" * 60)
    print("初始化完成! 下一步:")
    print("  python3 data_preprocessing.py --data_dir data/raw --output_dir data/processed")
    print("=" * 60)


if __name__ == "__main__":
    main()
