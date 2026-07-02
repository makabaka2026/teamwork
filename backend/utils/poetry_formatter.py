"""诗词格式化 — 清理生成文本，按行分割"""
import re


class PoetryFormatter:
    # 中文标点
    PUNCTUATION = set("，。！？；、：""''（）《》【】…—·")

    def format(self, raw_text: str, num_lines: int = 4) -> list[str]:
        """
        将原始生成文本格式化为诗词行列表
        """
        # 去除特殊 token
        text = self._clean(raw_text)

        # 按常见标点分割
        lines = self._split_lines(text)

        # 清理每行
        cleaned = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # 确保行尾有标点
            if line[-1] not in self.PUNCTUATION:
                line += "，"
            cleaned.append(line)

        # 截取需要的行数
        if len(cleaned) < num_lines:
            cleaned = cleaned[:num_lines]
        else:
            cleaned = cleaned[:num_lines]

        return cleaned

    def _clean(self, text: str) -> str:
        """去除特殊标记和多余空白"""
        # 去除 [CLS], [SEP], [PAD] 等
        text = re.sub(r"\[.*?\]", "", text)
        # 去除多余空白
        text = re.sub(r"\s+", "", text)
        # 只保留中文字符和中文标点
        text = re.sub(r"[^一-鿿　-〿＀-￯]", "", text)
        return text

    def _split_lines(self, text: str) -> list[str]:
        """按标点分割成行"""
        # 在句号、问号、感叹号、分号处断行
        text = re.sub(r"([。！？；])", r"\1\n", text)
        # 每 5-7 个逗号也断一下（处理连续句子）
        text = re.sub(r"((?:[^，。]*?，){2})", r"\1\n", text)
        return [l for l in text.split("\n") if l.strip()]

    def to_vertical(self, lines: list[str]) -> str:
        """转换为竖排显示（HTML）"""
        html = '<div class="poem-vertical">'
        for line in lines:
            html += f'<div class="poem-line">{line}</div>'
        html += "</div>"
        return html

    def join(self, lines: list[str]) -> str:
        """拼接为完整显示文本"""
        return "\n".join(lines)
