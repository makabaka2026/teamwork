"""诗词生成 API 路由"""
import json
import time
import traceback
import uuid
from datetime import datetime
from pathlib import Path

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
        _generator.load_model("gpt2")
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
    temperature = float(data.get("temperature", 0.7))
    model_type = data.get("model_type", "gpt2")

    try:
        start = time.time()
        gen = _get_generator()
        if model_type != gen.model_type:
            gen.switch_model(model_type)

        # 补全关键词到 2-3 字，给模型更好的起点
        prompt = _fix_prompt(keyword)
        raw_poem = gen.generate(prompt, num_lines=num_lines, temperature=temperature)
        poem_lines = _formatter.format(raw_poem, num_lines=num_lines)

        # 质量检测：噪声则降级
        if _poor(raw_poem):
            gen.switch_model("lstm")
            raw_poem = gen.generate(keyword, num_lines=num_lines, temperature=temperature)
            poem_lines = _formatter.format(raw_poem, num_lines=num_lines)

        rhyme_score = _rhyme_checker.score(poem_lines)
        elapsed = int((time.time() - start) * 1000)

        response_data = {
            "success": True,
            "poem": poem_lines,
            "keyword": keyword,
            "rhyme_score": round(rhyme_score, 2),
            "model": gen.model_type,
            "generation_time_ms": elapsed,
        }
        try:
            record = _build_history_record(
                "keyword", keyword, response_data,
                style=data.get("style", "tang"),
                num_lines=num_lines,
            )
            _save_history(record)
        except Exception:
            pass  # 保存失败不影响生成结果
        return jsonify(response_data)
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

        response_data = {
            "success": True,
            "poem": poem_lines,
            "extracted_keywords": keywords,
            "sentiment": sentiment,
            "rhyme_score": round(rhyme_score, 2),
            "model": gen.model_type,
            "generation_time_ms": elapsed,
        }
        try:
            record = _build_history_record(
                "scene", scene_text, response_data,
                style="tang", num_lines=num_lines,
            )
            _save_history(record)
        except Exception:
            pass
        return jsonify(response_data)
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

        response_data = {
            "success": True,
            "poem": poem_lines,
            "image_description": description,
            "recognized_labels": labels,
            "rhyme_score": round(rhyme_score, 2),
            "model": gen.model_type,
            "generation_time_ms": elapsed,
        }
        try:
            record = _build_history_record(
                "image", description, response_data,
                style="tang", num_lines=num_lines,
            )
            _save_history(record)
        except Exception:
            pass
        return jsonify(response_data)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


# ─── 历史记录 API ──────────────────────────────────────────

@api_bp.route("/history", methods=["GET"])
def get_history():
    """获取历史记录列表（时间倒序，支持分页）"""
    try:
        offset = request.args.get("offset", 0, type=int)
        limit = request.args.get("limit", 20, type=int)
        limit = min(limit, 100)  # 单次最多 100 条

        records = _load_history()
        total = len(records)
        items = records[offset:offset + limit]

        return jsonify({
            "success": True,
            "total": total,
            "has_more": offset + limit < total,
            "items": items,
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@api_bp.route("/history/<record_id>", methods=["DELETE"])
def delete_history(record_id: str):
    """删除单条历史记录"""
    try:
        records = _load_history()
        before = len(records)
        records = [r for r in records if r.get("id") != record_id]
        if len(records) == before:
            return jsonify({"success": False, "error": "记录不存在"}), 404

        tmp_path = HISTORY_FILE.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        tmp_path.replace(HISTORY_FILE)

        return jsonify({"success": True, "message": "已删除"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ─── 辅助函数 ───────────────────────────────────────────

def _fix_prompt(kw: str) -> str:
    """给关键词补常用搭配，让首行达到 5 或 7 字"""
    # 常用诗歌搭配，键为关键词，值为合适的扩展
    starters = {
        "山": "山色", "月": "月明", "春": "春风", "雪": "雪满",
        "花": "花开", "江": "江水", "柳": "柳丝", "云": "云深",
        "风": "风起", "梦": "梦回", "秋": "秋来", "夜": "夜半",
        "日": "日落", "雨": "雨过", "梅": "梅花", "竹": "竹影",
        "松": "松声", "鸟": "鸟啼", "人": "人生", "天": "天地",
        "水": "水流", "草": "草色", "露": "露华", "霜": "霜寒",
        "烟": "烟波", "舟": "舟行", "马": "马蹄", "剑": "剑气",
        "酒": "酒醒", "歌": "歌声", "楼": "楼台", "窗": "窗含",
        "门": "门前", "路": "路遥", "海": "海阔", "林": "林深",
    }
    return starters.get(kw, kw)


def _poor(raw: str) -> bool:
    for w in ["《", "》", "卷", "注", "本作", "按：", "项校", "〖", "〗"]:
        if w in raw:
            return True
    return len(raw) < 4


def _build_scene_prompt(keywords: list, sentiment: dict, scene_text: str) -> str:
    mood = sentiment.get("mood", "")
    kw_str = "、".join(keywords[:3])
    return f"关键词：{kw_str}。意境：{mood}。作诗一首：{scene_text}"


def _build_image_prompt(labels: list, description: str) -> str:
    kw_str = "、".join(labels[:3])
    return f"画面：{description}。关键词：{kw_str}。作诗一首。"


# ─── 历史记录存储 ───────────────────────────────────────────

HISTORY_DIR = Path(__file__).resolve().parent.parent / "data"
HISTORY_FILE = HISTORY_DIR / "history.json"
MAX_HISTORY = 500


def _init_history_dir():
    """确保 data 目录存在"""
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    return HISTORY_DIR


def _load_history() -> list:
    """加载历史记录，文件不存在或损坏时返回空列表"""
    try:
        if HISTORY_FILE.exists():
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
    except (json.JSONDecodeError, OSError):
        pass
    return []


def _build_history_record(gen_type: str, input_text: str, resp: dict, **extra) -> dict:
    """从生成响应构建历史记录对象"""
    return {
        "id": uuid.uuid4().hex[:8],
        "type": gen_type,
        "input": input_text,
        "poem": resp.get("poem", []),
        "style": extra.get("style", "tang"),
        "num_lines": resp.get("num_lines", len(resp.get("poem", []))),
        "rhyme_score": resp.get("rhyme_score", 0),
        "model": resp.get("model", "unknown"),
        "generation_time_ms": resp.get("generation_time_ms", 0),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def _save_history(record: dict) -> None:
    """保存一条新记录到历史文件头部，保留最近 500 条"""
    _init_history_dir()
    records = _load_history()
    records.insert(0, record)  # 最新在前
    if len(records) > MAX_HISTORY:
        records = records[:MAX_HISTORY]
    # 原子写入：先写临时文件再替换
    tmp_path = HISTORY_FILE.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    tmp_path.replace(HISTORY_FILE)
