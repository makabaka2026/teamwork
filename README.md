# 📜 诗词自动生成系统

基于深度学习的唐诗宋词自动生成 Web 应用，支持关键字生成、场景描述生成、图片识别生成三种模式。

## ✨ 功能

| 模式 | 输入 | 说明 |
|------|------|------|
| 🔑 关键字生成 | 一个中文关键词 | 基于 GPT-2 自动生成符合韵律的诗词 |
| 📝 场景生成 | 一段场景描述 | 文本分析 + 情感分析 → 生成诗词 |
| 🖼️ 图片生成 | 上传一张图片 | 图像识别 → 中文标签 → 生成诗词 |

## 🛠️ 技术栈

- **后端**：Flask + PyTorch + HuggingFace Transformers
- **前端**：Bootstrap 5 + Vanilla JavaScript
- **模型**：GPT2-Chinese-Poem / LSTM / Chinese-CLIP
- **NLP**：Jieba 分词 + SnowNLP 情感分析 + pypinyin 韵律校验

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 启动应用

```bash
python run.py
```

浏览器访问 `http://localhost:5000`

### 3. 下载模型（可选，首次启动自动下载）

```bash
# 国内用户建议设置 HuggingFace 镜像
set HF_ENDPOINT=https://hf-mirror.com
```

## 📁 项目结构

```
teamwork/
├── backend/                # 后端 Flask 应用
│   ├── app.py              # 应用入口
│   ├── config.py           # 配置
│   ├── routes/             # API 路由
│   ├── services/           # 核心服务（生成/分析/识别）
│   └── utils/              # 工具（韵律/格式化）
├── frontend/               # 前端页面
│   ├── templates/          # HTML 模板
│   └── static/             # CSS / JS
├── training/               # 模型训练脚本
├── requirements.txt        # Python 依赖
├── run.py                  # 一键启动
├── DEVELOPMENT.md          # 开发文档
└── 开发计划.md              # 开发计划
```

## 📡 API 接口

| 方法 | 路径 | 功能 |
|------|------|------|
| GET | `/api/health` | 健康检查 |
| POST | `/api/generate/keyword` | 关键字生成诗词 |
| POST | `/api/generate/scene` | 场景生成诗词 |
| POST | `/api/generate/image` | 图片生成诗词 |

详细文档见 [DEVELOPMENT.md](./DEVELOPMENT.md)

## 📊 开发进度

- [x] Sprint 1：项目骨架 + Flask 后端 + 前端页面
- [ ] Sprint 2：GPT-2 微调 + 韵律校验
- [ ] Sprint 3：NLP 管道 + 场景生成
- [ ] Sprint 4：图像识别 + 图片生成
- [ ] Sprint 5：前端美化 + 测试 + 文档

## 📝 License

MIT
