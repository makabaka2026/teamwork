# Sprint 2 开发报告 — BART 诗词生成系统

## 日期
2026-07-04

## 概述
将诗词生成系统从 GPT-2 重构为 BART Encoder-Decoder 架构，解决关键词相关性和格式一致性问题。

---

## 一、架构变更

### 之前 (GPT-2 Causal LM)
```
输入 "春风" → 直接拼接为诗句开头 → 关键词强制出现在首句
```
- 问题1: 关键词总是出现在诗句第一个字
- 问题2: 首行缺 1 个字 (2字关键词 + 模型生成2字 = 4字, 五言需要5字)
- 问题3: 五言/七言格式串扰

### 之后 (BART Encoder-Decoder)
```
Encoder: 关键词 "春风" → 语义编码
Decoder: 从零生成诗句 → 关键词不直接出现在首句
```
- Encoder 编码主题语义, Decoder 生成诗句, 天然分离
- 行字数由格式标签控制 (五言/七言)

---

## 二、数据预处理

### 清洗内容
1. **繁体→简体**: 180,600 处转换, 使用 zhconv 库
2. **残缺补字**: 移除 [X] 括号标记和 □ 占位符
3. **古籍注释**: 移除 `(一作XX)`, `(见《XX》卷X)` 等引用
4. **格式分类**: 五言绝句/律诗, 七言绝句/律诗, 宋词

### 训练数据格式 (Seq2Seq)
```json
{"input": "五言：春望", "output": "国破山河在，城春草木深。..."}
{"input": "五言：国破山河在，城春草木深。", "output": "国破山河在，..."}
```

两种模式:
- **标题→全诗**: 教会模型主题→诗映射
- **首联→全诗**: 教会模型场景→完整诗, 保持主题连贯

### 数据统计
| 文件 | 数量 | 大小 |
|------|------|------|
| seq2seq_tang.jsonl | 9,518 对 | 2.7 MB |
| seq2seq_wuyan.jsonl | 5,561 对 | 1.6 MB |
| seq2seq_qiyan.jsonl | 3,957 对 | 1.1 MB |
| seq2seq_songci.jsonl | 76,684 对 | ~20 MB |

---

## 三、模型训练

### 训练参数
- 基座模型: `fnlp/bart-base-chinese` (140M 参数)
- 优化器: AdamW, lr=5e-5
- Batch size: 4
- FP16: 开启
- GPU: NVIDIA RTX 4090 (24GB)

### 训练历史
| 版本 | 数据 | Epochs | Train Loss | 说明 |
|------|------|--------|------------|------|
| v1 | 5,283 对 (关键词) | 1 | 3.10 | 首行缺字已修复 |
| v2 | 5,283 对 (关键词) | 10 | 1.42 | 过拟合, 关键词关联弱 |
| v3 | 9,518 对 (标题+首联) | 5 | ~2.0 | 关键词关联改善 |
| v4 | 9,518 对 (标题+首联) | 5 | ~2.0 | 最终版本 |

---

## 四、关键技术方案

### 1. 关键词不在诗句开头
- 数据构建时跳过首句前4字与关键词重合的 pair
- 优先使用诗题 (如"春望""登高") 作为输入
- Encoder-Decoder 天然分离输入/输出

### 2. 首联扩写机制
```
用户输入: "春风"
    ↓ 检索首联索引 (8,086个关键词)
找到首联: "南陌春风早，东邻旭日新。"
    ↓ 拼接格式标签
输入模型: "五言：南陌春风早，东邻旭日新。"
    ↓ BART 继续生成
输出: 完整五言诗
```
首联索引从真实诗歌中提取, 保证了扩写的诗意质量。

### 3. 五言/七言格式控制
- 训练时添加格式标签: `"五言：xxx"` / `"七言：xxx"`
- 一个模型同时学两种格式, 共享情感/语法/词汇
- 格式标签作为条件控制输出行字数

---

## 五、文件变更清单

### 新建文件
| 文件 | 说明 |
|------|------|
| `training/train_bart.py` | BART Seq2Seq 训练脚本 |
| `training/setup_data.py` | 数据集初始化 (解压/解码文件名) |
| `training/data/processed/*.jsonl` | Seq2Seq 训练数据 |

### 修改文件
| 文件 | 变更 |
|------|------|
| `training/data_preprocessing.py` | 全面重写: 繁简转换、数据清洗、Seq2Seq输出、首联提取 |
| `backend/services/poetry_generator.py` | GPT2LMHeadModel → AutoModelForSeq2SeqLM (BART) |
| `backend/config.py` | 模型路径更新为 BART 系列 |
| `training/evaluate.py` | 适配 Seq2Seq 评估 |

### 本地文件 (未上传)
| 文件 | 说明 |
|------|------|
| `server.py` | 精简版 Web 服务器 (Flask + BART + 首联扩写) |
| `demo.py` | 命令行演示脚本 |
| `models/bart-tang/` | 训练好的 BART 模型 (~537MB) |
| `models/firstline_index.json` | 首联扩写索引 |
| `convert_simplified.py` | 批量繁→简转换脚本 |
| `build_expander.py` | 扩写词库构建脚本 |

---

## 六、待完成事项

1. [ ] **五言/七言分开训练** — 两个独立模型彻底解决格式串扰
2. [ ] **宋词模型训练** — 使用 76,684 对宋词数据
3. [ ] **故事→诗** — 长文本输入的诗意总结能力
4. [ ] **部署优化** — 模型量化、API 文档、生产环境配置
5. [ ] **韵律增强** — 押韵评分、平仄约束

---

## 七、服务器信息

- 地址: `connect.nmb1.seetacloud.com:44632`
- GPU: RTX 4090 (24GB)
- 项目路径: `/root/autodl-tmp/teamwork/`
- Python: `/root/miniconda3/bin/python3`
- HuggingFace 镜像: `HF_ENDPOINT=https://hf-mirror.com`
