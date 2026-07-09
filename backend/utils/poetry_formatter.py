"""诗词格式化 — 清理、按标点分行、繁简转换"""
import re
import zhconv


class PoetryFormatter:

    def format(self, raw_text: str, num_lines: int = 4) -> list[str]:
        """格式化生成文本为诗词行"""
        text = self._clean(raw_text)
        text = zhconv.convert(text, "zh-cn")
        return self._split_lines(text, num_lines)

    def _clean(self, text: str) -> str:
        text = re.sub(r"\[.*?\]", "", text)
        text = re.sub(r"\s+", "", text)
        return text

    def _split_lines(self, text: str, num_lines: int = 4) -> list[str]:
        """按标点自然分行：每个，。！？处断句"""
        result = []
        buf = ""
        for ch in text:
            buf += ch
            if ch in "。！？；，、":
                line = buf.strip()
                if len(line) >= 2:
                    result.append(line)
                buf = ""

        # 剩余内容
        if buf.strip() and len(buf.strip()) >= 2:
            result.append(buf.strip() + "。")

        # 如果标点分出的行不够（模型没生成足够标点）
        if len(result) < 2 and len(text) >= 8:
            chunk = 7 if len(text) >= 28 else 5
            result = []
            for i in range(0, len(text), chunk):
                seg = text[i:i+chunk]
                if len(seg) >= 3:
                    punct = "。" if (len(result) + 1) % 2 == 0 else "，"
                    result.append(seg + punct)

        return result[:num_lines]

    def to_vertical(self, lines: list[str]) -> str:
        html = '<div class="poem-vertical">'
        for line in lines:
            html += f'<div class="poem-line">{line}</div>'
        html += "</div>"
        return html

    def join(self, lines: list[str]) -> str:
        return "\n".join(lines)
