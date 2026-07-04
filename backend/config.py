"""应用配置"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "poetry-generator-secret-key")
    MAX_CONTENT_LENGTH = 10 * 1024 * 1024
    JSON_AS_ASCII = False

    CHECKPOINT_DIR = ROOT / "backend" / "checkpoints"

    # BART 模型 — 本地路径
    _MODEL_BASE = ROOT.parent / "models"
    BART_TANG = _MODEL_BASE / "bart-tang"

    @classmethod
    def get_model_dir(cls, style):
        return {
            "tang": cls.BART_TANG,
            "song": cls.BART_SONG,
            "wuyan": cls.BART_WUYAN,
            "qiyan": cls.BART_QIYAN,
        }.get(style, cls.BART_TANG)

    # 生成参数
    DEFAULT_NUM_LINES = 4
    DEFAULT_TEMPERATURE = 0.8

    # 训练数据
    TRAINING_DIR = ROOT / "training"
    PROCESSED_DATA = TRAINING_DIR / "data" / "processed"
