"""韵脚约束解码 — 在韵脚位置强制只从押韵字采样，其余位置自由生成"""
import re
import torch
from collections import defaultdict
from pypinyin import pinyin, Style


def _build_rhyme_groups() -> dict[str, set[int]]:
    """构建韵母 → 汉字集合映射，用于约束解码时快速查找"""
    # 诗词常用字表（覆盖绝大多数韵脚字）
    poem_chars = (
        # 高频韵脚字（按韵母分组）
        "天明清风平生情人心中深音襟林阴吟寻临今金禁琴心新尘身真门闻"
        "春风空中红同东通宫穷工雄鸿容龙重峰踪封浓钟从松逢梦送众痛洞"
        "安寒难看残叹阑干欢端宽酸团冠官盘瞒满半断乱岸暗叹"
        "光长霜茫伤肠觞忘香凉乡阳樯黄窗床量"
        "声轻清晴情鸣行惊争城成程明命"
        "流秋愁舟楼游洲浮收休州留求头柔柳酒久手九有后口走"
        "时思诗知离辞期枝迟丝池之"
        "归飞微衣稀非依扉晖违归肥机几希"
        "花开怀猜哀材栽海在台来待"
        "云分闻君文群军纷纷"
        "间山还关颜闲斑艰环弯前边天先川烟然仙弦年田连"
        "雪月绝别血蝶烈叶节切洁灭缺夜"
        "路处暮素故苦度步住数雨语去许女绿曲句玉足"
        "花草道老好早造噪告高柳笑小少妙照调"
        "落寞薄昨索诺"
        "客得色策北国白百"
        "美醉水会贵味费"
        "忆一日起西夕里几齐"
        "歌曲玉绿续烛"
        "山林晚看见间还关颜闲湾",
    )

    rhyme_dict = defaultdict(set)
    # 加载常见汉字的 token id (用 pypinyin)
    from transformers import AutoTokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained("fnlp/bart-base-chinese")
    except Exception:
        # 尝试本地加载
        import sys
        from pathlib import Path
        ROOT = Path(__file__).resolve().parent.parent.parent
        tokenizer = AutoTokenizer.from_pretrained(str(ROOT / "models" / "bart-qiyan"))

    for ch in set(poem_chars):
        if not ("一" <= ch <= "鿿"):
            continue
        try:
            py = pinyin(ch, style=Style.FINALS, strict=False)
            final = py[0][0] if py and py[0] else ""
            if final and len(final) >= 1:
                token_ids = tokenizer.encode(ch, add_special_tokens=False)
                for tid in token_ids:
                    rhyme_dict[final].add(tid)
        except Exception:
            pass

    # 合并相似韵母（放宽押韵标准）
    _merge_map = {
        "ong": ["ong", "eng", "iong"],
        "eng": ["eng", "ong"],
        "ing": ["ing", "in"],
        "in": ["in", "ing"],
        "en": ["en", "un", "in"],
        "un": ["un", "en"],
        "an": ["an", "ian", "uan"],
        "ian": ["ian", "an", "uan"],
        "uan": ["uan", "an", "ian"],
        "ang": ["ang", "iang", "uang"],
        "iang": ["iang", "ang", "uang"],
        "uang": ["uang", "ang", "iang"],
        "ao": ["ao", "iao"],
        "iao": ["iao", "ao"],
        "ai": ["ai", "uai"],
        "uai": ["uai", "ai"],
        "ei": ["ei", "ui", "uei"],
        "ui": ["ui", "ei", "uei"],
        "ou": ["ou", "iu", "iou"],
        "iu": ["iu", "ou", "iou"],
    }

    expanded = defaultdict(set)
    for final, ids in rhyme_dict.items():
        merged_ids = set(ids)
        if final in _merge_map:
            for related in _merge_map[final]:
                merged_ids.update(rhyme_dict.get(related, set()))
        # 也添加进原 final 的 key
        expanded[final] = merged_ids

    # 合并回原 rhyme_dict
    for final, ids in expanded.items():
        rhyme_dict[final] = ids

    return dict(rhyme_dict)


class RhymeConstrainedLogitsProcessor:
    """在韵脚位置限制采样范围，确保押韵"""

    def __init__(self, tokenizer, chars_per_line: int, num_lines: int,
                 rhyme_final: str = None, rhyme_dict: dict = None):
        """
        chars_per_line: 每句汉字数 (5或7)
        num_lines: 总行数 (4或8)
        rhyme_final: 目标韵母，不指定则从第2行自动检测
        """
        self.tokenizer = tokenizer
        self.chars_per_line = chars_per_line
        self.num_lines = num_lines
        self.rhyme_final = rhyme_final  # 待生成后确定
        self.rhyme_dict = rhyme_dict or _build_rhyme_groups()
        self._generated_pos = 0  # 当前已生成的 token 位置
        self._line_char_count = 0  # 当前行已生成汉字数
        self._current_line = 0  # 当前行号 (0-based)

        # 计算韵脚位置的 token offset（近似）
        # 每行: chars 个汉字 + 1 个标点
        self._tokens_per_line = chars_per_line + 1  # 近似

    def __call__(self, input_ids, scores):
        """在每个生成步调用，对韵脚位置的 logits 加约束"""
        # 如果还没确定韵母，先不做约束（自由生成前两行）
        if self.rhyme_final is None:
            # 等待生成足够多 token 后检测韵母
            return scores

        # 判断当前位置是否为韵脚位置
        rhyme_positions = self._get_rhyme_positions()

        if self._generated_pos in rhyme_positions:
            # 获取允许的 token ids（押韵字）
            allowed_ids = self.rhyme_dict.get(self.rhyme_final, set())
            if allowed_ids:
                # 创建 mask: 不押韵的 token 设为 -inf
                mask = torch.ones_like(scores) * float('-inf')
                for tid in allowed_ids:
                    if tid < scores.shape[-1]:
                        mask[:, tid] = scores[:, tid]
                # 保留原始概率分布，只过滤
                return mask

        return scores

    def _get_rhyme_positions(self) -> set[int]:
        """计算哪些生成位置是韵脚（偶数行最后一个汉字）"""
        rhyme_pos = set()
        tpl = self._tokens_per_line
        # 偶数行 (index 1, 3, 5, 7) 的最后一个汉字
        for line_idx in range(1, self.num_lines, 2):
            # 该行最后一个字的位置
            pos = line_idx * tpl + self.chars_per_line - 1
            rhyme_pos.add(pos)
        return rhyme_pos

    def advance_position(self):
        """每生成一个 token 后前进位置计数"""
        self._generated_pos += 1

    def set_rhyme_final(self, final: str):
        """设置目标韵母"""
        self.rhyme_final = final

    @staticmethod
    def detect_rhyme_final(text: str, chars_per_line: int = 7, target_line: int = 1) -> str:
        """从已生成文本中检测韵母（从第 target_line 行末字提取）"""
        chars = re.findall(r'[一-鿿]', text)
        # 第 target_line 行的最后一个字
        pos = (target_line + 1) * chars_per_line - 1  # target_line is 0-based
        if pos < len(chars):
            try:
                py = pinyin(chars[pos], style=Style.FINALS, strict=False)
                return py[0][0] if py and py[0] else ""
            except Exception:
                pass
        return ""
