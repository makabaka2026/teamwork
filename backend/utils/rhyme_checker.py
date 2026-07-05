"""韵律校验 — 基于 pypinyin 的押韵评分"""
from pypinyin import pinyin, Style


class RhymeChecker:
    def score(self, poem_lines: list[str]) -> float:
        """
        对诗词押韵程度打分 (0.0 - 1.0)
        规则：检查偶行（第2、4行）最后一个字的韵母是否相同
        """
        if len(poem_lines) < 2:
            return 0.5

        # 提取每行最后一个字的韵母
        finals = []
        for line in poem_lines:
            line = line.strip().rstrip("，、。！？；、.,!?;")
            if not line:
                continue
            final = self._get_final(line[-1])
            if final:
                finals.append(final)

        if len(finals) < 2:
            return 0.5

        # 偶行押韵得分
        even_finals = finals[1::2]  # 第2、4行
        if len(even_finals) < 2:
            # 只有4句时看第2和第4
            if len(finals) >= 4:
                f2, f4 = finals[1], finals[3]
                return 1.0 if self._is_rhyme(f2, f4) else 0.0
            return 0.5

        # 统计偶行之间韵母相同的比例
        match_count = 0
        total_pairs = 0
        for i in range(len(even_finals) - 1):
            for j in range(i + 1, len(even_finals)):
                if self._is_rhyme(even_finals[i], even_finals[j]):
                    match_count += 1
                total_pairs += 1

        return match_count / max(total_pairs, 1)

    def _get_final(self, char: str) -> str:
        """获取单个汉字的韵母"""
        try:
            py = pinyin(char, style=Style.FINALS, strict=False)
            return py[0][0] if py and py[0] else ""
        except Exception:
            return ""

    def _is_rhyme(self, final1: str, final2: str) -> bool:
        """判断两个韵母是否押韵（宽松匹配）"""
        if not final1 or not final2:
            return False
        # 完全相同
        if final1 == final2:
            return True
        # 忽略介母 (i/u/ü) 后比较
        import re
        core1 = re.sub(r"^[iuü]", "", final1)
        core2 = re.sub(r"^[iuü]", "", final2)
        return core1 == core2

    def check_rhythm(self, poem_lines: list[str]) -> dict:
        """
        检查平仄（简化版）
        返回每行字数和平仄模式
        """
        results = []
        for line in poem_lines:
            line = line.strip().rstrip("，、。！？；、.,!?;")
            chars = [c for c in line if "一" <= c <= "鿿"]
            if not chars:
                results.append({"text": line, "chars": 0, "pattern": ""})
                continue
            # 获取声调
            tones = []
            for ch in chars:
                try:
                    py = pinyin(ch, style=Style.TONE3, strict=False)
                    tone = py[0][0][-1] if py and py[0] else ""
                    tones.append(tone)
                except Exception:
                    tones.append("")
            results.append({
                "text": line,
                "chars": len(chars),
                "tones": tones,
            })
        return {"lines": results}
