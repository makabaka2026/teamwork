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
                document.getElementById("labels-preview").innerHTML =
                    `<span class="badge bg-info">识别: ${data.recognized_labels.join(", ")}</span>`;
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
        extraHtml += `图像识别: ${data.recognized_labels.join("、")} | `;
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

function showLoading(show) {
    document.getElementById("loading-spinner").style.display = show ? "block" : "none";
    document.querySelectorAll("button.btn-primary").forEach(btn => {
        btn.disabled = show;
        btn.innerHTML = show
            ? '<span class="spinner-border spinner-border-sm me-1"></span>生成中...'
            : btn.dataset.origText || btn.innerHTML;
    });
    if (!show) {
        document.querySelectorAll("button.btn-primary").forEach(btn => {
            if (btn.dataset.origText) btn.innerHTML = btn.dataset.origText;
        });
    } else {
        document.querySelectorAll("button.btn-primary").forEach(btn => {
            btn.dataset.origText = btn.innerHTML;
        });
    }
}

function showError(msg) {
    const alert = document.getElementById("error-alert");
    alert.textContent = msg.startsWith("✅") ? "📋 " + msg.substring(2).trim() : "⚠️ " + msg;
    alert.style.display = "block";
    alert.className = "alert mt-3 " + (msg.startsWith("✅") ? "alert-success" : "alert-danger");
    setTimeout(() => alert.style.display = "none", 3000);
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

// ─── 历史 & 浏览 ────────────────────────────────────

let historyState = {
    data: [],           // 全部已加载数据
    offset: 0,
    hasMore: false,
    loaded: false,      // 是否已首次加载
    currentFilter: "all",
};

let browseState = {
    data: [],
    offset: 0,
    hasMore: false,
    loaded: false,
    currentFilter: "all",
    currentModalItem: null,
};

const HISTORY_PAGE_SIZE = 20;

// ─── 加载 ──────────────────────────────────────────

async function loadHistoryIfNeeded() {
    if (!historyState.loaded) {
        await fetchHistory("history");
    }
}

async function loadBrowseIfNeeded() {
    if (!browseState.loaded) {
        await fetchHistory("browse");
    }
}

async function fetchHistory(mode) {
    const state = mode === "history" ? historyState : browseState;
    try {
        const resp = await fetch(`/api/history?offset=${state.offset}&limit=${HISTORY_PAGE_SIZE}`);
        const data = await resp.json();
        if (data.success) {
            state.data = state.data.concat(data.items);
            state.hasMore = data.has_more;
            state.offset += data.items.length;
            state.loaded = true;
            if (mode === "history") {
                renderHistoryList();
            } else {
                renderBrowseCards();
            }
        }
    } catch (e) {
        showError("加载历史失败: " + e.message);
    }
}

function loadMoreHistory() {
    fetchHistory("history");
}

function loadMoreBrowse() {
    fetchHistory("browse");
}

// ─── 筛选 ──────────────────────────────────────────

function applyHistoryFilter() {
    historyState.currentFilter = document.getElementById("history-filter").value;
    renderHistoryList();
}

function applyBrowseFilter() {
    browseState.currentFilter = document.getElementById("browse-filter").value;
    renderBrowseCards();
}

function filterItems(items, filterType) {
    if (filterType === "all") return items;
    return items.filter(item => item.type === filterType);
}

// ─── 历史列表视图 ───────────────────────────────────

function renderHistoryList() {
    const container = document.getElementById("history-list");
    const empty = document.getElementById("history-empty");
    const loadMore = document.getElementById("history-load-more");

    const filtered = filterItems(historyState.data, historyState.currentFilter);

    if (historyState.data.length === 0) {
        container.innerHTML = "";
        empty.style.display = "block";
        loadMore.style.display = "none";
        return;
    }

    empty.style.display = filtered.length === 0 ? "block" : "none";

    const typeLabels = { keyword: "🔑", scene: "📝", image: "🖼️" };

    container.innerHTML = filtered.map(item => `
        <div class="history-item border rounded p-3 mb-2">
            <div class="d-flex justify-content-between align-items-start">
                <div class="flex-grow-1">
                    <div class="d-flex align-items-center gap-2 mb-1">
                        <span class="badge bg-secondary">${typeLabels[item.type] || item.type} ${item.type}</span>
                        <small class="text-muted">${item.timestamp}</small>
                        <small class="text-muted">${item.model}</small>
                        <span class="rhyme-stars small">${starRating(item.rhyme_score)}</span>
                    </div>
                    <div class="text-muted small mb-1">输入: ${escapeHtml(item.input)}</div>
                    <div class="history-preview">${escapeHtml(item.poem.slice(0, 2).join(" "))}</div>
                </div>
                <div class="d-flex gap-1 flex-shrink-0 ms-2">
                    <button class="btn btn-outline-secondary btn-sm" onclick="copyHistoryPoem('${item.id}', this)" title="复制">📋</button>
                    <button class="btn btn-outline-danger btn-sm" onclick="deleteHistoryItem('${item.id}')" title="删除">🗑️</button>
                </div>
            </div>
        </div>
    `).join("");

    loadMore.style.display = historyState.hasMore ? "block" : "none";
}

function starRating(score) {
    const n = Math.round(score * 5);
    return "★".repeat(n) + "☆".repeat(5 - n);
}

// ─── 卡片画廊视图 ──────────────────────────────────

function renderBrowseCards() {
    const container = document.getElementById("browse-cards");
    const empty = document.getElementById("browse-empty");
    const loadMore = document.getElementById("browse-load-more");

    const filtered = filterItems(browseState.data, browseState.currentFilter);

    if (browseState.data.length === 0) {
        container.innerHTML = "";
        empty.style.display = "block";
        loadMore.style.display = "none";
        return;
    }

    empty.style.display = filtered.length === 0 ? "block" : "none";

    const typeLabels = { keyword: "🔑 关键字", scene: "📝 场景", image: "🖼️ 图片" };

    container.innerHTML = filtered.map(item => `
        <div class="col-lg-4 col-md-6">
            <div class="card poem-card h-100" onclick="showPoemModal('${item.id}')" style="cursor:pointer;">
                <div class="card-body d-flex flex-column">
                    <div class="d-flex justify-content-between align-items-start mb-2">
                        <span class="badge bg-primary">${typeLabels[item.type] || item.type}</span>
                        <span class="rhyme-stars small">${starRating(item.rhyme_score)}</span>
                    </div>
                    <p class="card-text poem-card-text flex-grow-1">${escapeHtml(item.poem.slice(0, 2).join("\n"))}</p>
                    <div class="text-muted small mt-2">
                        <span>${escapeHtml(item.input.substring(0, 15))}${item.input.length > 15 ? "…" : ""}</span>
                    </div>
                </div>
            </div>
        </div>
    `).join("");

    loadMore.style.display = browseState.hasMore ? "block" : "none";
}

// ─── 模态框 ────────────────────────────────────────

function showPoemModal(id) {
    const item = browseState.data.find(r => r.id === id);
    if (!item) return;
    browseState.currentModalItem = item;

    document.getElementById("poem-modal-title").textContent =
        `${item.type === "keyword" ? "🔑" : item.type === "scene" ? "📝" : "🖼️"} ${item.input.substring(0, 20)}`;
    document.getElementById("poem-modal-body").innerHTML =
        item.poem.map(line => `<span class="line">${escapeHtml(line)}</span>`).join("\n");
    document.getElementById("poem-modal-meta").innerHTML =
        `生成时间: ${item.timestamp} | 模型: ${item.model} | 韵律评分: ${starRating(item.rhyme_score)} (${item.rhyme_score}) | 耗时: ${(item.generation_time_ms / 1000).toFixed(1)}s`;

    new bootstrap.Modal(document.getElementById("poem-modal")).show();
}

function copyModalPoem() {
    if (!browseState.currentModalItem) return;
    const text = browseState.currentModalItem.poem.join("\n");
    navigator.clipboard.writeText(text).then(() => {
        showError("✅ 已复制到剪贴板");
    });
}

// ─── 历史列表操作 ──────────────────────────────────

function copyHistoryPoem(id, btn) {
    const item = historyState.data.find(r => r.id === id);
    if (!item) return;
    const text = item.poem.join("\n");
    navigator.clipboard.writeText(text).then(() => {
        const orig = btn.textContent;
        btn.textContent = "✅";
        setTimeout(() => btn.textContent = orig, 1500);
    });
}

async function deleteHistoryItem(id) {
    if (!confirm("确定要删除这条记录吗？")) return;
    try {
        const resp = await fetch(`/api/history/${id}`, { method: "DELETE" });
        const data = await resp.json();
        if (data.success) {
            historyState.data = historyState.data.filter(r => r.id !== id);
            browseState.data = browseState.data.filter(r => r.id !== id);
            renderHistoryList();
            renderBrowseCards();
        } else {
            showError(data.error || "删除失败");
        }
    } catch (e) {
        showError("删除失败: " + e.message);
    }
}
