"""文本分析 — Jieba 分词 + TF-IDF/TextRank 关键词提取"""
import jieba
import jieba.analyse
import jieba.posseg as pseg


class TextAnalyzer:
    def __init__(self):
        # 加载自定义词典（诗词相关术语）
        poetry_words = ["五言", "七言", "绝句", "律诗", "词牌", "押韵", "平仄"]
        for w in poetry_words:
            jieba.add_word(w)

    def segment(self, text: str) -> list[tuple[str, str]]:
        """分词并标注词性"""
        return list(pseg.cut(text))

    def extract_keywords(self, text: str, top_n: int = 5) -> list[str]:
        """提取关键词（TextRank 算法）"""
        keywords = jieba.analyse.textrank(
            text, topK=top_n, withWeight=False, allowPOS=("n", "nr", "ns", "a", "v")
        )
        if not keywords:
            # 降级：TF-IDF
            keywords = jieba.analyse.extract_tags(text, topK=top_n)
        return keywords

    def extract_keywords_tfidf(self, text: str, top_n: int = 5) -> list[str]:
        """TF-IDF 关键词提取"""
        return jieba.analyse.extract_tags(text, topK=top_n)

    def get_keywords_with_weights(self, text: str, top_n: int = 5) -> list[tuple[str, float]]:
        """带权重的关键词"""
        return jieba.analyse.textrank(
            text, topK=top_n, withWeight=True,
        )
