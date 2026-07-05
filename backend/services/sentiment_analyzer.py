"""情感分析 — SnowNLP 情感极性判断"""
from snownlp import SnowNLP


class SentimentAnalyzer:
    # 情感 → 诗词意境的映射
    MOOD_MAP = {
        "positive": "明快豪放",
        "negative": "婉约忧愁",
        "neutral": "淡雅闲适",
    }

    def analyze(self, text: str) -> dict:
        """
        分析文本情感
        返回: {"label": "positive/negative/neutral", "score": 0.0-1.0, "mood": "诗词意境标签"}
        """
        s = SnowNLP(text)
        score = s.sentiments  # 0-1, 越接近 1 越正面

        if score > 0.6:
            label = "positive"
        elif score < 0.4:
            label = "negative"
        else:
            label = "neutral"

        return {
            "label": label,
            "score": round(score, 4),
            "mood": self.MOOD_MAP[label],
        }

    def get_mood_tags(self, text: str) -> list[str]:
        """返回多个可能的意境标签"""
        s = SnowNLP(text)
        score = s.sentiments

        tags = []
        if score > 0.6:
            tags.extend(["豪放", "明快", "喜悦"])
        elif score < 0.4:
            tags.extend(["婉约", "忧愁", "伤感"])
        else:
            tags.extend(["淡雅", "闲适", "宁静"])

        # 分析关键词补充标签
        nature_words = {"春", "花", "鸟", "月", "山", "水", "柳", "风", "雨", "雪", "云"}
        for w in self._tokenize(text):
            if w in nature_words:
                tags.append("写景")
                break

        return list(dict.fromkeys(tags))  # 去重保序

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        import jieba
        return list(jieba.cut(text))
