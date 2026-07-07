"""应用配置"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "poetry-generator-secret-key")
    MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10MB 上传限制
    JSON_AS_ASCII = False  # 支持中文 JSON

    # 模型路径
    CHECKPOINT_DIR = ROOT / "backend" / "checkpoints"
    GPT2_MODEL_DIR = CHECKPOINT_DIR / "gpt2-poem"
    LSTM_MODEL_PATH = CHECKPOINT_DIR / "lstm" / "lstm_tang.pth"
    CLIP_MODEL_DIR = CHECKPOINT_DIR / "clip"

    # 生成参数默认值
    DEFAULT_NUM_LINES = 4
    DEFAULT_TEMPERATURE = 0.8
    DEFAULT_STYLE = "tang"

    # 训练数据路径
    TRAINING_DATA_DIR = ROOT / "training" / "data"
    RAW_DATA_DIR = TRAINING_DATA_DIR / "raw"
    PROCESSED_DATA_DIR = TRAINING_DATA_DIR / "processed"
