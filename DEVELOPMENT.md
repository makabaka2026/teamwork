# 诗词自动生成系统 - 开发文档

## 一、项目概述

基于深度学习的唐诗宋词自动生成系统，支持关键字生成、场景描述生成、图片识别生成三种模式。

- **技术栈**：Flask + PyTorch + HuggingFace Transformers
- **模型**：GPT2-Chinese-Poem（主力）、LSTM（备选）、Chinese-CLIP（图像识别）
- **平台**：Web 应用（Bootstrap 5 响应式布局）

---

## 二、项目结构

```
teamwork/
├── backend/                          # 后端
│   ├── app.py                        # Flask 应用工厂
│   ├── config.py                     # 全局配置
│   ├── routes/
│   │   └── poetry_routes.py          # REST API 路由（3个端点 + 健康检查）
│   ├── services/
│   │   ├── poetry_generator.py       # 诗词生成核心（GPT-2/LSTM）
│   │   ├── lstm_model.py             # LSTM 网络定义
│   │   ├── text_analyzer.py          # Jieba 分词 + TextRank 关键词提取
│   │   ├── sentiment_analyzer.py     # SnowNLP 情感分析
│   │   └── image_recognizer.py       # Chinese-CLIP 图像识别
│   └── utils/
│       ├── rhyme_checker.py          # pypinyin 韵律校验
│       └── poetry_formatter.py       # 输出格式化
├── frontend/                         # 前端
│   ├── templates/
│   │   ├── base.html                 # 基础布局（Bootstrap 5）
│   │   └── index.html               # 主页面（三Tab切换）
│   └── static/
│       ├── css/style.css             # 自定义样式
│       └── js/main.js                # 前端交互逻辑
├── training/                         # 训练脚本
│   ├── data_preprocessing.py         # 数据预处理
│   ├── train_lstm.py                 # LSTM 训练
│   ├── train_gpt2.py                 # GPT-2 微调
│   └── evaluate.py                   # 模型评估
├── tests/                            # 单元测试
├── requirements.txt                  # Python 依赖
├── run.py                            # 一键启动入口
└── .gitignore                        # Git 忽略规则
```

---

## 三、API 接口文档

### 3.1 健康检查
```
GET /api/health
→ {"status": "ok", "model": "gpt2", "gpu_available": false}
```

### 3.2 关键字生成诗词 (F1)
```
POST /api/generate/keyword
Content-Type: application/json

请求：
{
    "keyword": "春",          // 必填，1-10个中文字符
    "num_lines": 4,           // 可选，4或8（默认4）
    "style": "tang",          // 可选，"tang"唐诗 / "song"宋词
    "temperature": 0.8,       // 可选，0.1-1.5（默认0.8）
    "model_type": "gpt2"      // 可选，"gpt2" / "lstm"
}

响应：
{
    "success": true,
    "poem": ["春眠不觉晓，", "处处闻啼鸟。", ...],
    "keyword": "春",
    "rhyme_score": 0.92,
    "model": "gpt2",
    "generation_time_ms": 1234
}
```

### 3.3 场景生成诗词 (F2)
```
POST /api/generate/scene
Content-Type: application/json

请求：
{
    "scene_text": "春天江边柳树摇曳，心情愉快",
    "num_lines": 4,
    "temperature": 0.8
}

响应：
{
    "success": true,
    "poem": [...],
    "extracted_keywords": ["春天", "江边", "柳树"],
    "sentiment": {"label": "positive", "score": 0.87, "mood": "明快豪放"},
    "rhyme_score": 0.88,
    "generation_time_ms": 2100
}
```

### 3.4 图片生成诗词 (F3)
```
POST /api/generate/image
Content-Type: multipart/form-data

表单：
    image: <图片文件>        // jpg/png/gif/webp，≤10MB
    num_lines: 4
    temperature: 0.8

响应：
{
    "success": true,
    "poem": [...],
    "image_description": "山水、孤舟、落日",
    "recognized_labels": ["山水", "孤舟", "落日", ...],
    "rhyme_score": 0.85,
    "generation_time_ms": 2800
}
```

---

## 四、环境搭建

### 4.1 安装依赖

```bash
cd teamwork
python -m venv .venv
.venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

### 4.2 HuggingFace 国内镜像（可选）

```bash
set HF_ENDPOINT=https://hf-mirror.com      # Windows
# export HF_ENDPOINT=https://hf-mirror.com  # Unix
```

### 4.3 下载数据集

```bash
cd training/data/raw
git clone --depth 1 https://github.com/chinese-poetry/chinese-poetry.git
# 将 json/tang/ 移到 training/data/raw/tang/
```

### 4.4 数据预处理

```bash
cd training
python data_preprocessing.py --format both
```

### 4.5 训练模型

```bash
# LSTM
python train_lstm.py --epochs 50

# GPT-2
python train_gpt2.py --epochs 3
```

### 4.6 启动应用

```bash
python run.py
# 访问 http://localhost:5000
```

---

## 五、模型说明

| 模型 | 用途 | 大小 | 来源 |
|------|------|------|------|
| GPT2-Chinese-Poem | 诗词生成主力 | ~180MB | HuggingFace `uer/gpt2-chinese-poem` |
| LSTM | 诗词生成备选 | ~50MB | 自训练 |
| Chinese-CLIP | 图像识别 | ~600MB | HuggingFace `OFA-Sys/chinese-clip-vit-base-patch16` |
| Jieba + TextRank | 关键词提取 | ~5MB | pip install jieba |
| SnowNLP | 情感分析 | ~10MB | pip install snownlp |
| pypinyin | 韵律校验 | ~2MB | pip install pypinyin |

### 降级策略

- GPT-2 加载失败 → 自动切换到 LSTM
- Chinese-CLIP 不可用 → 使用轻量颜色特征分类
- 所有模型未训练 → 使用内置回退诗词库

---

## 六、开发路线图

| Sprint | 内容 | 状态 |
|--------|------|------|
| Sprint 1 | 项目骨架 + Flask 后端 + 前端页面 | ✅ 完成 |
| Sprint 2 | GPT-2 微调 + 韵律校验集成 | ⏳ |
| Sprint 3 | NLP 管道 + 场景生成 (F2) | ⏳ |
| Sprint 4 | 图像识别 + 图片生成 (F3) | ⏳ |
| Sprint 5 | 前端美化 + 测试 + 文档 | ⏳ |

---

## 七、Git 使用

```bash
# 如果使用代理
git config --global http.proxy http://127.0.0.1:51081
git config --global https.proxy http://127.0.0.1:51081

# 提交
git add -A
git commit -m "[Sprint N] 描述"
git push origin main
```
