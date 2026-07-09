"""图像识别 — BLIP 自由描述 + MarianMT 英译中 → 中文关键词 → 写诗"""
import jieba.analyse
import torch
from PIL import Image
from backend.config import Config


class ImageRecognizer:
    def __init__(self):
        self.model_type = "none"
        # BLIP
        self.blip_processor = None
        self.blip_model = None
        # MarianMT 翻译
        self.trans_tokenizer = None
        self.trans_model = None
        # CLIP 降级
        self.clip_model = None
        self.clip_processor = None

    def load_model(self):
        """BLIP + MarianMT，失败降级到 CLIP，再失败降到轻量方案"""
        # 1. BLIP
        try:
            self._load_blip()
        except Exception as e:
            print(f"[WARN] BLIP 加载失败 ({e})")

        # 2. MarianMT
        try:
            self._load_translator()
        except Exception as e:
            print(f"[WARN] 翻译模型加载失败 ({e})")

        # 判断使用哪条路径
        if self.blip_model and self.trans_model:
            self.model_type = "blip_translate"
            print("[INFO] 图片识别: BLIP + MarianMT (自由描述模式)")
        elif self.blip_model:
            self.model_type = "blip_only"
            print("[INFO] 图片识别: BLIP (英文描述)")
        else:
            # 降级 CLIP
            try:
                self._load_clip()
                self.model_type = "chinese-clip"
                print("[INFO] 图片识别: CLIP (标签分类, 降级)")
            except Exception as e:
                print(f"[WARN] CLIP 加载失败 ({e})，使用轻量方案")
                self.model_type = "lightweight"

    def _load_blip(self):
        from transformers import BlipProcessor, BlipForConditionalGeneration
        model_name = "Salesforce/blip-image-captioning-base"
        self.blip_processor = BlipProcessor.from_pretrained(model_name, local_files_only=True)
        self.blip_model = BlipForConditionalGeneration.from_pretrained(model_name, local_files_only=True)
        self.blip_model.eval()

    def _load_translator(self):
        from transformers import MarianMTModel, MarianTokenizer
        path = str(Config.MARIAN_MODEL_DIR)
        self.trans_tokenizer = MarianTokenizer.from_pretrained(path, local_files_only=True)
        self.trans_model = MarianMTModel.from_pretrained(path, local_files_only=True)
        self.trans_model.eval()

    def _load_clip(self):
        from transformers import ChineseCLIPProcessor, ChineseCLIPModel
        model_name = "OFA-Sys/chinese-clip-vit-base-patch16"
        self.clip_model = ChineseCLIPModel.from_pretrained(model_name, local_files_only=True)
        self.clip_processor = ChineseCLIPProcessor.from_pretrained(model_name, local_files_only=True)
        self.clip_model.eval()

    def recognize(self, image_path: str) -> tuple[list[str], str]:
        if self.model_type in ("blip_translate", "blip_only"):
            return self._recognize_blip(image_path)
        elif self.model_type == "chinese-clip":
            return self._recognize_clip(image_path)
        else:
            return self._recognize_lightweight(image_path)

    def _recognize_blip(self, image_path: str) -> tuple[list[str], str]:
        """BLIP 描述 → 翻译 → jieba 关键词"""
        image = Image.open(image_path).convert("RGB")

        # 1. BLIP 英文描述
        inputs = self.blip_processor(image, return_tensors="pt")
        with torch.no_grad():
            out = self.blip_model.generate(**inputs, max_new_tokens=50)
        en_caption = self.blip_processor.decode(out[0], skip_special_tokens=True)
        print(f"[BLIP] {en_caption}")

        # 2. 翻译（如果有翻译模型）
        if self.trans_model:
            zh_caption = self._translate(en_caption)
        else:
            zh_caption = en_caption  # 降级：直接用英文
        print(f"[ZH] {zh_caption}")

        # 3. jieba 提取中文关键词（过滤停用词）
        _stop_words = {
            "一幅", "一个", "这个", "那个", "一些", "这种", "那种",
            "的", "了", "是", "在", "和", "与", "或", "及",
            "有", "很", "都", "也", "就", "还", "要", "能", "会",
            "可以", "可能", "应该", "没有", "不是", "因为", "所以",
            "但", "而", "从", "到", "对", "把", "被", "让", "给",
            "中", "里", "上", "下", "前", "后", "左", "右",
            "这", "那", "它", "他", "她", "们", "着", "过",
            "不", "没", "来", "去", "说", "看", "想", "做",
            "用", "为", "以", "所", "其",
        }
        keywords = [w for w in jieba.analyse.extract_tags(zh_caption, topK=8)
                    if w not in _stop_words and len(w) >= 2][:5]
        if not keywords:
            # 降级：直接分词
            words = [w for w in jieba.cut(zh_caption)
                    if len(w) >= 2 and w not in _stop_words][:5]
            keywords = words
        if not keywords:
            keywords = ["风景", "自然"]

        description = zh_caption if zh_caption else en_caption
        return keywords, description

    def _translate(self, en_text: str) -> str:
        """英→中翻译"""
        if len(en_text) > 200:
            en_text = en_text[:200]
        inputs = self.trans_tokenizer(en_text, return_tensors="pt", padding=True,
                                       truncation=True, max_length=128)
        with torch.no_grad():
            translated = self.trans_model.generate(
                **inputs, max_new_tokens=128,
                num_beams=4, early_stopping=True,
            )
        return self.trans_tokenizer.decode(translated[0], skip_special_tokens=True)

    def _recognize_clip(self, image_path: str) -> tuple[list[str], str]:
        """CLIP 标签分类（降级路径）"""
        labels = [
            "巍峨高山","潺潺溪流","辽阔大海","金色沙滩","静谧湖泊",
            "飞瀑流泉","云雾缭绕","茫茫雪原","冰川雪山","火山熔岩",
            "蓝天白云","绚丽晚霞","雨后彩虹","乌云密布","电闪雷鸣",
            "皓月当空","繁星满天","旭日东升","夕阳西下","大雾迷蒙",
            "春日桃花","夏日荷花","秋日红叶","冬日腊梅","翠绿竹林",
            "花间蝴蝶","枝头小鸟","水中游鱼","骏马奔腾","仙鹤独立",
            "古典园林","亭台楼阁","寺庙佛塔","江南水乡","小桥流水",
            "繁华都市","霓虹夜景","幽静小巷","热闹集市","茶馆书院",
            "思乡怀远","离愁别绪","欢聚畅饮","孤独寂寞","悠然自得",
            "大漠孤烟","边塞烽火","金戈铁马","戍守边关","塞外风雪",
        ]
        image = Image.open(image_path).convert("RGB")
        inputs = self.clip_processor(text=labels, images=image,
                                      return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = self.clip_model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1)
        top_idx = probs[0].argsort(descending=True)[:5].tolist()
        top_labels = [labels[i] for i in top_idx]
        description = "、".join(top_labels[:3])
        return top_labels, description

    def _recognize_lightweight(self, image_path: str) -> tuple[list[str], str]:
        """颜色启发式（终极降级）"""
        import numpy as np
        image = Image.open(image_path).convert("RGB")
        img = image.resize((64, 64))
        pixels = np.array(img, dtype=np.float32)
        r, g, b = pixels.mean(axis=(0, 1))
        brightness = (r + g + b) / 3
        labels = []
        if brightness > 150:
            labels.append("旭日东升")
        elif brightness < 80:
            labels.append("繁星满天")
        else:
            labels.append("山水风景")
        if g > r + 15 and g > b + 15:
            labels.append("翠绿竹林")
        elif b > r + 15 and b > g + 15:
            labels.append("辽阔大海")
        elif r > g + 15 and r > b + 15:
            labels.append("绚丽晚霞")
        labels.append("田园乡村")
        return labels[:5], "、".join(labels[:3])
