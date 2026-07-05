"""图像识别 — Chinese-CLIP 图片→中文描述"""
import torch
from PIL import Image
from backend.config import Config


# 预设中文标签（用于轻量级降级方案）
_SCENE_LABELS = [
    "山水风景", "花鸟虫鱼", "明月星空", "离愁别绪",
    "边塞战场", "田园乡村", "宫廷楼阁", "江南水乡",
    "大漠孤烟", "竹林深处", "雪景寒梅", "春日桃李",
    "秋叶黄花", "江边垂柳", "渔舟唱晚", "归雁南飞",
    "孤舟独钓", "长亭送别", "松风明月", "荷塘月色",
]


class ImageRecognizer:
    def __init__(self):
        self.model = None
        self.processor = None
        self.model_type = "none"

    def load_model(self):
        """加载 Chinese-CLIP，失败则使用轻量降级方案"""
        try:
            self._load_clip()
            self.model_type = "chinese-clip"
        except Exception as e:
            print(f"[WARN] Chinese-CLIP 加载失败 ({e})，使用轻量方案")
            self.model_type = "lightweight"

    def _load_clip(self):
        """加载 Chinese-CLIP 模型"""
        from transformers import ChineseCLIPProcessor, ChineseCLIPModel

        model_name = "OFA-Sys/chinese-clip-vit-base-patch16"
        print(f"[INFO] 加载 Chinese-CLIP: {model_name} ...")

        self.model = ChineseCLIPModel.from_pretrained(model_name)
        self.processor = ChineseCLIPProcessor.from_pretrained(model_name)
        self.model.eval()
        print("[INFO] Chinese-CLIP 加载完成")

    def recognize(self, image_path: str) -> tuple[list[str], str]:
        """
        识别图片内容
        返回: (标签列表, 中文描述)
        """
        if self.model_type == "chinese-clip":
            return self._recognize_clip(image_path)
        else:
            return self._recognize_lightweight(image_path)

    def _recognize_clip(self, image_path: str) -> tuple[list[str], str]:
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(
            text=_SCENE_LABELS, images=image,
            return_tensors="pt", padding=True,
        )

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)

        # 取 top-5 标签
        top_indices = probs[0].argsort(descending=True)[:5].tolist()
        labels = [_SCENE_LABELS[i] for i in top_indices]
        description = "、".join(labels[:3])
        return labels, description

    def _recognize_lightweight(self, image_path: str) -> tuple[list[str], str]:
        """轻量方案：基于图像颜色/亮度特征做简单分类"""
        from PIL import Image
        import numpy as np

        image = Image.open(image_path).convert("RGB")
        img = image.resize((64, 64))
        pixels = np.array(img, dtype=np.float32)

        # 简单特征
        avg_color = pixels.mean(axis=(0, 1))  # 平均 RGB
        brightness = avg_color.mean()
        saturation = np.std(pixels, axis=(0, 1)).mean()

        # 根据颜色/亮度给标签
        labels = []
        r, g, b = avg_color

        if brightness > 150:
            labels.append("春日桃李")
        elif brightness < 80:
            labels.append("雪景寒梅")
        else:
            labels.append("山水风景")

        if g > r and g > b:
            labels.append("竹林深处")
        if b > r and b > g:
            labels.append("荷塘月色")
        if r > g and r > b and brightness > 120:
            labels.append("边塞战场")

        labels.append("田园乡村")  # 兜底
        labels = labels[:5]
        description = "、".join(labels[:3])
        return labels, description
