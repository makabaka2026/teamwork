/* 诗词自动生成系统 - 前端交互 */

let lastRequest = null; // 保存上一次请求参数用于重新生成

// ─── 初始化 ────────────────────────────────────────

document.addEventListener("DOMContentLoaded", () => {
    setupUploadArea();
    checkHealth();
});

// ─── 健康检查 ──────────────────────────────────────

async function checkHealth() {
    try {
        const res = await fetch("/api/health");
        const data = await res.json();
        console.log("Backend status:", data);
    } catch (e) {
        console.warn("Backend not ready:", e);
    }
}

// ─── F1: 关键字生成 ────────────────────────────────

async function generateKeyword() {
    const keyword = document.getElementById("keyword-input").value.trim();
    if (!keyword) return showError("请输入关键字");

    const params = {
        keyword: keyword,
        num_lines: parseInt(document.getElementById("keyword-lines").value),
        style: document.getElementById("keyword-style").value,
        model_type: document.getElementById("keyword-model").value,
        temperature: parseFloat(document.getElementById("keyword-temp").value),
    };

    lastRequest = { type: "keyword", params };
    await doGenerate("/api/generate/keyword", "POST", params);
}

// ─── F2: 场景生成 ──────────────────────────────────

async function generateScene() {
    const sceneText = document.getElementById("scene-input").value.trim();
    if (!sceneText) return showError("请输入场景描述");

    const params = {
        scene_text: sceneText,
        num_lines: parseInt(document.getElementById("scene-lines").value),
        temperature: parseFloat(document.getElementById("scene-temp").value),
    };

    lastRequest = { type: "scene", params };
    await doGenerate("/api/generate/scene", "POST", params, (data) => {
        // 显示分析结果
        if (data.extracted_keywords) {
            document.getElementById("analysis-preview").innerHTML =
                `<span class="badge bg-info me-1">关键词: ${data.extracted_keywords.join(", ")}</span>` +
                `<span class="badge bg-secondary">情感: ${data.sentiment?.mood || "—"}</span>`;
        }
    });
}

// ─── F3: 图片生成 ──────────────────────────────────

async function generateImage() {
    const fileInput = document.getElementById("image-file");
    const file = fileInput.files[0];
    if (!file) return showError("请先选择图片");

    const formData = new FormData();
    formData.append("image", file);
    formData.append("num_lines", document.getElementById("image-lines").value);
    formData.append("temperature", document.getElementById("image-temp").value);

    lastRequest = { type: "image", formData };
    showLoading(true);
    hideResult();
    hideError();

    try {
        const res = await fetch("/api/generate/image", { method: "POST", body: formData });
        const data = await res.json();
        showLoading(false);

        if (data.success) {
            showResult(data);
            if (data.recognized_labels) {
                const filtered = _filterLabels(data.recognized_labels);
                document.getElementById("labels-preview").innerHTML =
                    `<span class="badge bg-info">识别: ${filtered.join(", ")}</span>`;
            }
        } else {
            showError(data.error);
        }
    } catch (e) {
        showLoading(false);
        showError("请求失败: " + e.message);
    }
}

// ─── 通用生成请求 ──────────────────────────────────

async function doGenerate(url, method, body, extraCallback) {
    showLoading(true);
    hideResult();
    hideError();

    try {
        const res = await fetch(url, {
            method: method,
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
        });
        const data = await res.json();
        showLoading(false);

        if (data.success) {
            showResult(data);
            if (extraCallback) extraCallback(data);
        } else {
            showError(data.error);
        }
    } catch (e) {
        showLoading(false);
        showError("请求失败: " + e.message);
    }
}

// ─── 显示结果 ──────────────────────────────────────

function showResult(data) {
    const card = document.getElementById("result-card");
    const display = document.getElementById("poem-display");
    const rhymeInfo = document.getElementById("rhyme-info");
    const timeInfo = document.getElementById("time-info");
    const extraInfo = document.getElementById("extra-info");

    // 渲染诗句
    const lines = data.poem.map(line =>
        `<span class="line">${escapeHtml(line)}</span>`
    ).join("\n");
    display.innerHTML = lines;

    // 韵律评分
    const stars = "★".repeat(Math.round(data.rhyme_score * 5)) +
                  "☆".repeat(5 - Math.round(data.rhyme_score * 5));
    rhymeInfo.innerHTML = `韵律评分: <span class="rhyme-stars">${stars}</span> (${data.rhyme_score})`;

    // 耗时
    timeInfo.textContent = `生成耗时: ${(data.generation_time_ms / 1000).toFixed(1)}s`;

    // 额外信息
    let extraHtml = "";
    if (data.extracted_keywords) {
        extraHtml += `关键词: ${data.extracted_keywords.join("、")} | `;
    }
    if (data.sentiment) {
        extraHtml += `情感: ${data.sentiment.mood}(${data.sentiment.label}) | `;
    }
    if (data.recognized_labels) {
        const filtered = _filterLabels(data.recognized_labels);
        extraHtml += `图像识别: ${filtered.join("、")} | `;
    }
    extraHtml += `模型: ${data.model}`;
    extraInfo.innerHTML = extraHtml;

    card.style.display = "block";
    card.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

// ─── 重新生成 ──────────────────────────────────────

function regenerate() {
    if (!lastRequest) return;
    if (lastRequest.type === "keyword") generateKeyword();
    else if (lastRequest.type === "scene") generateScene();
    else if (lastRequest.type === "image") generateImage();
}

// ─── 复制诗词 ──────────────────────────────────────

function copyPoem() {
    const display = document.getElementById("poem-display");
    const text = Array.from(display.querySelectorAll(".line"))
        .map(el => el.textContent)
        .join("\n");

    navigator.clipboard.writeText(text).then(() => {
        const btn = event.target;
        const orig = btn.textContent;
        btn.textContent = "✅ 已复制";
        setTimeout(() => btn.textContent = orig, 2000);
    });
}

// ─── 图片上传 ──────────────────────────────────────

function setupUploadArea() {
    const area = document.getElementById("upload-area");
    const fileInput = document.getElementById("image-file");
    const preview = document.getElementById("image-preview");
    const placeholder = document.getElementById("upload-placeholder");

    // 点击上传
    area.addEventListener("click", () => fileInput.click());

    // 文件选择
    fileInput.addEventListener("change", () => {
        const file = fileInput.files[0];
        if (file) showPreview(file, preview, placeholder);
    });

    // 拖拽
    area.addEventListener("dragover", (e) => {
        e.preventDefault();
        area.classList.add("dragover");
    });

    area.addEventListener("dragleave", () => {
        area.classList.remove("dragover");
    });

    area.addEventListener("drop", (e) => {
        e.preventDefault();
        area.classList.remove("dragover");
        const file = e.dataTransfer.files[0];
        if (file) {
            fileInput.files = e.dataTransfer.files;
            showPreview(file, preview, placeholder);
        }
    });
}

function showPreview(file, preview, placeholder) {
    const reader = new FileReader();
    reader.onload = (e) => {
        preview.src = e.target.result;
        preview.classList.remove("d-none");
        placeholder.classList.add("d-none");
    };
    reader.readAsDataURL(file);
}

// ─── UI 辅助 ───────────────────────────────────────

// 每个按钮用固定的 data-key 标识
const _BTN_KEYS = ['keyword-btn', 'scene-btn', 'image-btn'];
const _btnOriginalText = {};

function showLoading(show) {
    const spinner = document.getElementById("loading-spinner");
    if (spinner) spinner.style.display = show ? "block" : "none";

    // 找到当前可见 tab 里的按钮
    const visiblePanel = document.querySelector('.tab-pane.active, .tab-pane.show.active');
    const btn = visiblePanel ? visiblePanel.querySelector('button.btn-primary') : null;

    if (!btn) return;

    // 用按钮所在 panel 的 id 做 key（唯一且不变）
    const panelId = visiblePanel.id || 'unknown';

    if (show) {
        if (!_btnOriginalText[panelId]) {
            _btnOriginalText[panelId] = btn.textContent.trim();
        }
        btn.disabled = true;
        btn.innerHTML = '<span class="spinner-border spinner-border-sm me-1"></span>生成中...';
    } else {
        btn.disabled = false;
        btn.innerHTML = _btnOriginalText[panelId] || btn.innerHTML;
    }
}

// 图片标签美化：过滤无意义停用词
function _filterLabels(labels) {
    const stop = new Set([
        '一幅', '一个', '这个', '那个', '一些', '这种', '那种',
        '的', '了', '是', '在', '和', '与', '或', '及',
        '有', '很', '都', '也', '就', '还', '要', '能', '会',
        '可以', '可能', '应该', '没有', '不是', '因为', '所以',
        '但', '而', '从', '到', '对', '把', '被', '让', '给',
        '中', '里', '上', '下', '前', '后', '左', '右',
    ]);
    return (labels || []).filter(l => !stop.has(l) && l.length >= 2);
}

function showError(msg) {
    const alert = document.getElementById("error-alert");
    alert.textContent = "⚠️ " + msg;
    alert.style.display = "block";
    setTimeout(() => alert.style.display = "none", 5000);
}

function hideError() {
    document.getElementById("error-alert").style.display = "none";
}

function hideResult() {
    document.getElementById("result-card").style.display = "none";
}

function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
}
