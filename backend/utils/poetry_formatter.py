"""诗词格式化 — 清理生成文本，按行分割，繁简转换"""
import re
import zhconv


class PoetryFormatter:
    PUNCTUATION = set("，。！？；、：""''（）《》【】…—·")

    def format(self, raw_text: str, num_lines: int = 4) -> list[str]:
        # 去除特殊 token
        text = self._clean(raw_text)
        # 繁体转简体
        text = zhconv.convert(text, "zh-cn")
        # 按标点分割
        lines = self._split_lines(text)
        # 清理每行
        cleaned = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line[-1] not in self.PUNCTUATION:
                line += "，"
            cleaned.append(line)

        return cleaned[:num_lines]

    def _clean(self, text: str) -> str:
        text = re.sub(r"\[.*?\]", "", text)
        text = re.sub(r"\s+", "", text)
        text = re.sub(r"[^一-鿿　-〿＀-￯]", "", text)
        return text

    def _split_lines(self, text: str) -> list[str]:
        text = re.sub(r"([。！？；])", r"\1\n", text)
        text = re.sub(r"((?:[^，。]*?，){2})", r"\1\n", text)
        return [l for l in text.split("\n") if l.strip()]

    def to_vertical(self, lines: list[str]) -> str:
        html = '<div class="poem-vertical">'
        for line in lines:
            html += f'<div class="poem-line">{line}</div>'
        html += "</div>"
        return html

    def join(self, lines: list[str]) -> str:
        return "\n".join(lines)
