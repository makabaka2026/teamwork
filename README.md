# 诗词自动生成系统

基于深度学习的唐诗宋词自动生成 Web 应用，支持关键字生成、场景描述生成、图片识别生成三种模式。

## 功能

| 模式 | 输入 | 说明 | 状态 |
|------|------|------|------|
| 关键字生成 | 一个中文关键词（1-10字） | 回退诗库匹配 → 生成符合韵律的诗词 | ✅ 可用 |
| 场景生成 | 一段场景描述 | Jieba 分词 + SnowNLP 情感分析 → 生成诗词 | ✅ 可用 |
| 图片生成 | 上传一张图片 | 色彩特征识别 → 中文标签 → 生成诗词 | ✅ 可用 |

## 技术栈

| 技术 | 版本 | 用途 |
|------|------|------|
| Python | 3.14.6 | 运行环境 |
| Flask | 3.1.3 | Web 框架 |
| PyTorch | 2.12.1+cpu | 深度学习框架（CPU 版） |
| Transformers | 5.12.1 | HuggingFace 模型加载（GPT-2 / CLIP） |
| Jieba | 0.42.1 | 中文分词 + TextRank 关键词提取 |
| SnowNLP | 0.12.3 | 中文情感分析 |
| pypinyin | 0.55.0 | 拼音转换 + 韵律校验 |
| zhconv | 1.4.3 | 繁体转简体 |
| OpenCV | 5.0.0.93 | 图像处理 |
| Pillow | 12.3.0 | 图片读写 |
| NumPy | 2.5.0 | 数值计算 |
| Bootstrap | 5.x | 前端 UI 框架（CDN） |
| Waitress | 3.0.2 | 生产环境 WSGI 服务器 |

## 数据来源

- **御定全唐诗**（chinese-poetry 仓库）：39689 首唐诗，用于构建回退诗库和 LSTM 词汇表
- 回退诗库覆盖 1695 个关键词，7297 个不重复汉字

## 快速开始

```bash
# 1. 创建虚拟环境
python -m venv .venv
source .venv/Scripts/activate   # Windows

# 2. 安装依赖（~3.5 GB，含 PyTorch）
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# 3. 设置 HuggingFace 国内镜像（加速模型下载）
export HF_ENDPOINT=https://hf-mirror.com

# 4. 启动应用
python run.py
```

浏览器访问 `http://localhost:5000`

## 项目结构

```
teamwork/
├── backend/
│   ├── app.py                     # Flask 应用工厂
│   ├── config.py                  # 全局配置
│   ├── checkpoints/               # 模型权重 + 回退诗库 + 词汇表
│   │   ├── fallback_poems.json    # 1695 关键词回退诗库
│   │   └── vocab.pkl              # 7297 字符词汇表
│   ├── routes/
│   │   └── poetry_routes.py       # 4 个 REST API 端点
│   ├── services/
│   │   ├── poetry_generator.py    # GPT-2 / LSTM 诗词生成核心
│   │   ├── lstm_model.py          # LSTM 网络定义（3层）
│   │   ├── text_analyzer.py       # Jieba 关键词提取
│   │   ├── sentiment_analyzer.py  # SnowNLP 情感分析
│   │   ├── image_recognizer.py    # CLIP / 轻量图像识别
│   │   └── build_fallback.py      # 回退诗库构建脚本
│   └── utils/
│       ├── rhyme_checker.py       # pypinyin 韵律评分
│       └── poetry_formatter.py    # 输出格式化 + 繁简转换
├── frontend/
│   ├── templates/
│   │   ├── base.html              # Bootstrap 5 基础布局
│   │   └── index.html             # 三 Tab 主页面
│   └── static/
│       ├── css/style.css
│       └── js/main.js             # AJAX 请求 + 图片上传
├── training/
│   ├── data_preprocessing.py      # 数据预处理（适配御定全唐诗格式）
│   ├── train_lstm.py              # LSTM 训练脚本
│   ├── train_gpt2.py              # GPT-2 微调脚本
│   ├── evaluate.py                # 模型评估（BLEU / Perplexity）
│   └── data/                      # 训练数据（gitignore）
├── requirements.txt               # 完整依赖
├── requirements-backend.txt       # 角色A：后端引擎
├── requirements-training.txt      # 角色B：模型训练
├── requirements-frontend.txt      # 角色C：前端开发
├── requirements-api.txt           # 角色D：API+测试
├── run.py                         # 一键启动入口
├── 团队分工与协作规范.md            # 分工 + 接口规范
├── 开发计划.md
└── 项目要求.md
```

## API 接口

| 方法 | 路径 | 功能 | 状态 |
|------|------|------|------|
| GET | `/api/health` | 健康检查（模型状态、GPU 可用性） | ✅ |
| POST | `/api/generate/keyword` | F1：关键字 → 诗词 | ✅ |
| POST | `/api/generate/scene` | F2：场景 → 分词+情感+诗词 | ✅ |
| POST | `/api/generate/image` | F3：图片 → 识别+诗词 | ✅ |

详细接口文档见 [团队分工与协作规范.md](./团队分工与协作规范.md)

## 当前进度

| Sprint | 内容 | 状态 |
|--------|------|------|
| Sprint 1 | 项目骨架 + Flask 后端 + 前端页面 | ✅ 完成 |
| Sprint 2 | 回退诗库 + 繁简转换 + API 全通 | ✅ 完成 |
| Sprint 3 | 模型训练（LSTM / GPT-2 微调） | ⏳ 待角色B |
| Sprint 4 | 前端美化 + 响应式适配 | ⏳ 待角色C |
| Sprint 5 | 单元测试 + 文档完善 | ⏳ 待角色D |

## 降级策略

当模型权重未训练好时，系统自动降级：
- GPT-2 下载失败 → 切换 LSTM
- LSTM 无训练权重 → 使用 1695 关键词回退诗库（39689 首唐诗）
- Chinese-CLIP 加载失败 → 使用色彩特征轻量识别
- 所有降级对前端透明，API 始终返回有效诗词

## License

MIT
