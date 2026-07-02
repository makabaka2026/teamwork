"""诗词生成 API 路由"""
import time
import traceback
from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename

from backend.services.poetry_generator import PoetryGenerator
from backend.services.text_analyzer import TextAnalyzer
from backend.services.sentiment_analyzer import SentimentAnalyzer
from backend.services.image_recognizer import ImageRecognizer
from backend.utils.rhyme_checker import RhymeChecker
from backend.utils.poetry_formatter import PoetryFormatter

api_bp = Blueprint("api", __name__)

# 懒加载模型（首次请求时初始化）
_generator: PoetryGenerator = None
_text_analyzer: TextAnalyzer = None
_sentiment_analyzer: SentimentAnalyzer = None
_image_recognizer: ImageRecognizer = None
_rhyme_checker = RhymeChecker()
_formatter = PoetryFormatter()

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "bmp", "webp"}


def _get_generator() -> PoetryGenerator:
    global _generator
    if _generator is None:
        _generator = PoetryGenerator()
        _generator.load_model()
    return _generator


def _get_text_analyzer() -> TextAnalyzer:
    global _text_analyzer
    if _text_analyzer is None:
        _text_analyzer = TextAnalyzer()
    return _text_analyzer


def _get_sentiment_analyzer() -> SentimentAnalyzer:
    global _sentiment_analyzer
    if _sentiment_analyzer is None:
        _sentiment_analyzer = SentimentAnalyzer()
    return _sentiment_analyzer


def _get_image_recognizer() -> ImageRecognizer:
    global _image_recognizer
    if _image_recognizer is None:
        _image_recognizer = ImageRecognizer()
        _image_recognizer.load_model()
    return _image_recognizer


def _allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ─── 健康检查 ───────────────────────────────────────────

@api_bp.route("/health", methods=["GET"])
def health():
    try:
        gen = _get_generator()
        return jsonify({
            "status": "ok",
            "model": gen.model_type,
            "gpu_available": gen.gpu_available,
        })
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 503


# ─── F1: 关键字生成诗词 ──────────────────────────────────

@api_bp.route("/generate/keyword", methods=["POST"])
def generate_keyword():
    data = request.get_json(silent=True) or {}
    keyword = data.get("keyword", "").strip()

    if not keyword:
        return jsonify({"success": False, "error": "请提供关键字"}), 400
    if len(keyword) > 10:
        return jsonify({"success": False, "error": "关键字不超过10个字符"}), 400

    num_lines = int(data.get("num_lines", 4))
    temperature = float(data.get("temperature", 0.8))
    model_type = data.get("model_type", "gpt2")

    try:
        start = time.time()
        gen = _get_generator()
        if model_type != gen.model_type:
            gen.switch_model(model_type)

        raw_poem = gen.generate(keyword, num_lines=num_lines, temperature=temperature)
        poem_lines = _formatter.format(raw_poem, num_lines=num_lines)
        rhyme_score = _rhyme_checker.score(poem_lines)
        elapsed = int((time.time() - start) * 1000)

        return jsonify({
            "success": True,
            "poem": poem_lines,
            "keyword": keyword,
            "rhyme_score": round(rhyme_score, 2),
            "model": gen.model_type,
            "generation_time_ms": elapsed,
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


# ─── F2: 场景生成诗词 ──────────────────────────────────

@api_bp.route("/generate/scene", methods=["POST"])
def generate_scene():
    data = request.get_json(silent=True) or {}
    scene_text = data.get("scene_text", "").strip()

    if not scene_text:
        return jsonify({"success": False, "error": "请提供场景描述"}), 400

    num_lines = int(data.get("num_lines", 4))
    temperature = float(data.get("temperature", 0.8))

    try:
        start = time.time()

        # 文本分析
        ta = _get_text_analyzer()
        keywords = ta.extract_keywords(scene_text, top_n=5)

        # 情感分析
        sa = _get_sentiment_analyzer()
        sentiment = sa.analyze(scene_text)

        # 构建 prompt 并生成
        prompt = _build_scene_prompt(keywords, sentiment, scene_text)
        gen = _get_generator()
        raw_poem = gen.generate(prompt, num_lines=num_lines, temperature=temperature)
        poem_lines = _formatter.format(raw_poem, num_lines=num_lines)
        rhyme_score = _rhyme_checker.score(poem_lines)
        elapsed = int((time.time() - start) * 1000)

        return jsonify({
            "success": True,
            "poem": poem_lines,
            "extracted_keywords": keywords,
            "sentiment": sentiment,
            "rhyme_score": round(rhyme_score, 2),
            "model": gen.model_type,
            "generation_time_ms": elapsed,
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


# ─── F3: 图片生成诗词 ──────────────────────────────────

@api_bp.route("/generate/image", methods=["POST"])
def generate_image():
    if "image" not in request.files:
        return jsonify({"success": False, "error": "请上传图片文件"}), 400

    file = request.files["image"]
    if file.filename == "" or not _allowed_file(file.filename):
        return jsonify({"success": False, "error": "不支持的图片格式，请上传 jpg/png/gif 等"}), 400

    num_lines = int(request.form.get("num_lines", 4))
    temperature = float(request.form.get("temperature", 0.8))

    try:
        start = time.time()

        # 保存临时文件
        import tempfile
        from pathlib import Path
        tmp_dir = Path(tempfile.gettempdir()) / "poetry_uploads"
        tmp_dir.mkdir(exist_ok=True)
        tmp_path = tmp_dir / secure_filename(file.filename)
        file.save(str(tmp_path))

        # 图像识别
        ir = _get_image_recognizer()
        labels, description = ir.recognize(str(tmp_path))

        # 清理临时文件
        tmp_path.unlink(missing_ok=True)

        # 构建 prompt 并生成
        prompt = _build_image_prompt(labels, description)
        gen = _get_generator()
        raw_poem = gen.generate(prompt, num_lines=num_lines, temperature=temperature)
        poem_lines = _formatter.format(raw_poem, num_lines=num_lines)
        rhyme_score = _rhyme_checker.score(poem_lines)
        elapsed = int((time.time() - start) * 1000)

        return jsonify({
            "success": True,
            "poem": poem_lines,
            "image_description": description,
            "recognized_labels": labels,
            "rhyme_score": round(rhyme_score, 2),
            "model": gen.model_type,
            "generation_time_ms": elapsed,
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


# ─── 辅助函数 ───────────────────────────────────────────

def _build_scene_prompt(keywords: list, sentiment: dict, scene_text: str) -> str:
    mood = sentiment.get("mood", "")
    kw_str = "、".join(keywords[:3])
    return f"关键词：{kw_str}。意境：{mood}。作诗一首：{scene_text}"


def _build_image_prompt(labels: list, description: str) -> str:
    kw_str = "、".join(labels[:3])
    return f"画面：{description}。关键词：{kw_str}。作诗一首。"
