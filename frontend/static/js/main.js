/* 诗词自动生成系统 - 前端交互 */

let lastRequest = null;

// ─── 风格切换 ──────────────────────────────────
function onStyleChange() {
    const style = document.getElementById("keyword-style").value;
    document.getElementById("tang-lines-group").style.display = style === "tang" ? "" : "none";
    document.getElementById("song-lines-group").style.display = style === "song" ? "" : "none";
}

// ─── 初始化 ────────────────────────────────────────

document.addEventListener("DOMContentLoaded", () => {
    setupUploadArea();
    checkHealth();
    initParallax();
    initScrollReveal();
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

    const style = document.getElementById("keyword-style").value;
    const numLines = style === "song"
        ? parseInt(document.getElementById("song-lines").value)
        : parseInt(document.getElementById("keyword-lines").value);

    const params = {
        keyword: keyword,
        num_lines: numLines,
        style: style,
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

    // 渲染诗句，每行带 index 用于逐行动画
    const lines = data.poem.map((line, i) =>
        `<span class="line" style="--line-index:${i}">${escapeHtml(line)}</span>`
    ).join("\n");
    display.innerHTML = lines;

    // 韵律评分
    const stars = "★".repeat(Math.round(data.rhyme_score * 5)) +
                  "☆".repeat(5 - Math.round(data.rhyme_score * 5));
    rhymeInfo.innerHTML = `韵律评分: <span class="rhyme-stars">${stars}</span> (${data.rhyme_score.toFixed(1)})`;

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

    // 在按钮上触发墨点粒子
    const btn = document.activeElement && document.activeElement.classList.contains('btn-primary')
        ? document.activeElement
        : document.querySelector("button.btn-primary:not([disabled])");
    if (btn) spawnInkParticles(btn);
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
        if (show) {
            btn.dataset.origText = btn.innerHTML;
            btn.innerHTML = '挥毫中…';
        } else {
            btn.innerHTML = btn.dataset.origText || btn.innerHTML;
        }
    });
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
    authors: [],      // 作者列表
    rhythmics: [],    // 词牌名列表
};

const HISTORY_PAGE_SIZE = 20;
const LIBRARY_PAGE_SIZE = 24;

// ─── 加载 ──────────────────────────────────────────

async function loadHistoryIfNeeded() {
    if (!historyState.loaded) {
        await fetchHistory();
    }
}

async function loadBrowseIfNeeded() {
    if (!browseState.loaded) {
        // 并行加载诗词库和筛选列表
        await Promise.all([
            fetchLibrary(),
            fetchLibraryFilters()
        ]);
    }
}

async function fetchHistory() {
    try {
        const resp = await fetch(`/api/history?offset=${historyState.offset}&limit=${HISTORY_PAGE_SIZE}`);
        const data = await resp.json();
        if (data.success) {
            historyState.data = historyState.data.concat(data.items);
            historyState.hasMore = data.has_more;
            historyState.offset += data.items.length;
            historyState.loaded = true;
            renderHistoryList();
        }
    } catch (e) {
        showError("加载历史失败: " + e.message);
    }
}

async function fetchLibrary(reset = false) {
    if (reset) {
        browseState.offset = 0;
        browseState.data = [];
    }
    const filter = browseState.currentFilter;
    let url = `/api/library?offset=${browseState.offset}&limit=${LIBRARY_PAGE_SIZE}`;
    if (filter === "all") {
        // 无筛选
    } else if (filter.startsWith("author:")) {
        url += `&author=${encodeURIComponent(filter.replace("author:", ""))}`;
    } else if (filter.startsWith("rhythmic:")) {
        url += `&rhythmic=${encodeURIComponent(filter.replace("rhythmic:", ""))}`;
    }

    try {
        const resp = await fetch(url);
        const data = await resp.json();
        if (data.success) {
            browseState.data = browseState.data.concat(data.items);
            browseState.hasMore = data.has_more;
            browseState.offset += data.items.length;
            browseState.loaded = true;
            renderBrowseCards();
        }
    } catch (e) {
        showError("加载诗词库失败: " + e.message);
    }
}

async function fetchLibraryFilters() {
    try {
        const [authorsRes, rhythmicsRes] = await Promise.all([
            fetch("/api/library/authors"),
            fetch("/api/library/rhythmics"),
        ]);
        const authorsData = await authorsRes.json();
        const rhythmicsData = await rhythmicsRes.json();
        if (authorsData.success) browseState.authors = authorsData.authors;
        if (rhythmicsData.success) browseState.rhythmics = rhythmicsData.rhythmics;
        renderBrowseFilter();
    } catch (e) {
        console.warn("加载筛选列表失败:", e);
    }
}

function renderBrowseFilter() {
    const select = document.getElementById("browse-filter");
    if (!select) return;
    let html = '<option value="all">全部诗词</option>';
    html += '<optgroup label="── 按作者 ──">';
    browseState.authors.forEach(a => {
        html += `<option value="author:${a}">${a}</option>`;
    });
    html += '</optgroup>';
    html += '<optgroup label="── 按词牌 ──">';
    browseState.rhythmics.forEach(r => {
        html += `<option value="rhythmic:${r}">${r}</option>`;
    });
    html += '</optgroup>';
    select.innerHTML = html;
}

function loadMoreHistory() {
    fetchHistory();
}

function loadMoreBrowse() {
    fetchLibrary();
}

// ─── 筛选 ──────────────────────────────────────────

function applyHistoryFilter() {
    historyState.currentFilter = document.getElementById("history-filter").value;
    renderHistoryList();
}

function applyBrowseFilter() {
    browseState.currentFilter = document.getElementById("browse-filter").value;
    // 重置分页，重新加载
    browseState.offset = 0;
    browseState.data = [];
    browseState.hasMore = false;
    renderBrowseCards();
    fetchLibrary(true);
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

    const items = browseState.data;

    if (items.length === 0 && browseState.loaded) {
        container.innerHTML = "";
        empty.style.display = "block";
        loadMore.style.display = "none";
        return;
    }

    empty.style.display = "none";

    container.innerHTML = items.map(item => `
        <div class="col-lg-3 col-md-4 col-sm-6">
            <div class="card poem-card h-100" onclick="showPoemModal('${item.id}')" style="cursor:pointer;">
                <div class="card-body d-flex flex-column">
                    <h6 class="poem-card-author">${escapeHtml(item.author)}</h6>
                    ${item.rhythmic ? `<span class="poem-card-rhythmic">【${escapeHtml(item.rhythmic)}】</span>` : ""}
                    <p class="card-text poem-card-text flex-grow-1">${escapeHtml(item.paragraphs.slice(0, 2).join("\n"))}</p>
                    <div class="text-muted small mt-2">
                        <span>共 ${item.paragraphs.length} 句</span>
                    </div>
                </div>
            </div>
        </div>
    `).join("");

    loadMore.style.display = browseState.hasMore ? "block" : "none";
}

// ─── 模态框 ────────────────────────────────────────

let poemModalInstance = null;

function closePoemModal() {
    if (poemModalInstance) {
        poemModalInstance.hide();
        // dispose 延迟到 hidden 事件后
    }
}

function showPoemModal(id) {
    const item = browseState.data.find(r => r.id === id);
    if (!item) return;
    browseState.currentModalItem = item;

    const isLibrary = item.author && item.paragraphs;

    if (isLibrary) {
        document.getElementById("poem-modal-title").textContent =
            `${item.author} · ${item.rhythmic || "无题"}`;
        document.getElementById("poem-modal-body").innerHTML =
            item.paragraphs.map((line, i) => `<span class="line" style="--line-index:${i}">${escapeHtml(line)}</span>`).join("\n");
        document.getElementById("poem-modal-meta").innerHTML =
            `作者: ${item.author} | 词牌: ${item.rhythmic || "—"} | 共 ${item.paragraphs.length} 句`;
    } else {
        document.getElementById("poem-modal-title").textContent =
            `${item.type === "keyword" ? "🔑" : item.type === "scene" ? "📝" : "🖼️"} ${(item.input || "").substring(0, 20)}`;
        document.getElementById("poem-modal-body").innerHTML =
            (item.poem || []).map((line, i) => `<span class="line" style="--line-index:${i}">${escapeHtml(line)}</span>`).join("\n");
        document.getElementById("poem-modal-meta").innerHTML =
            `生成时间: ${item.timestamp || "—"} | 模型: ${item.model || "—"} | 韵律评分: ${starRating(item.rhyme_score || 0)} (${(item.rhyme_score || 0).toFixed(1)})`;
    }

    // 先关闭旧实例
    if (poemModalInstance) {
        poemModalInstance.hide();
        poemModalInstance.dispose();
        poemModalInstance = null;
    }

    const modalEl = document.getElementById("poem-modal");
    poemModalInstance = new bootstrap.Modal(modalEl, {
        backdrop: true,
        keyboard: true,
        focus: true
    });
    poemModalInstance.show();

    // 监听关闭事件，清理实例
    modalEl.addEventListener("hidden.bs.modal", () => {
        if (poemModalInstance) {
            poemModalInstance.dispose();
            poemModalInstance = null;
        }
    }, { once: true });
}

function copyModalPoem() {
    if (!browseState.currentModalItem) return;
    const item = browseState.currentModalItem;
    const text = (item.paragraphs || item.poem || []).join("\n");
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

// ─── 墨点飞溅粒子效果 ──────────────────────────────────

function spawnInkParticles(button) {
    const rect = button.getBoundingClientRect();
    const cx = rect.left + rect.width / 2;
    const cy = rect.top + rect.height / 2;
    const count = 14;

    for (let i = 0; i < count; i++) {
        const particle = document.createElement("span");
        const angle = (Math.PI * 2 * i) / count;
        const distance = 30 + Math.random() * 40;
        const sx = Math.cos(angle) * distance;
        const sy = Math.sin(angle) * distance;
        const size = 3 + Math.random() * 6;

        Object.assign(particle.style, {
            position: "fixed",
            left: cx + "px",
            top: cy + "px",
            width: size + "px",
            height: size + "px",
            background: `rgba(${Math.random() > 0.5 ? "44,36,22" : "196,61,61"}, ${0.5 + Math.random() * 0.5})`,
            borderRadius: "50%",
            pointerEvents: "none",
            zIndex: "9999",
            transition: "all 0.6s ease-out",
        });

        document.body.appendChild(particle);

        requestAnimationFrame(() => {
            particle.style.transform = `translate(${sx}px, ${sy}px) scale(0)`;
            particle.style.opacity = "0";
        });

        setTimeout(() => particle.remove(), 700);
    }
}

// ─── 背景视差效果 ──────────────────────────────────────

function initParallax() {
    const mountains = document.querySelector(".bg-mountains");
    const clouds = document.querySelector(".bg-clouds");
    if (!mountains && !clouds) return;

    document.addEventListener("mousemove", (e) => {
        const x = (e.clientX / window.innerWidth - 0.5) * 2;  // -1 ~ 1
        const y = (e.clientY / window.innerHeight - 0.5) * 2;

        if (mountains) {
            mountains.style.transform = `translate(${x * 8}px, ${y * 4}px)`;
        }
        if (clouds) {
            clouds.style.transform = `translate(${x * -12}px, ${y * -6}px)`;
        }
    });
}

// ─── 滚动入场动画 ──────────────────────────────────────

function initScrollReveal() {
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.animation = "fadeInUp 0.5s ease-out forwards";
                observer.unobserve(entry.target);
            }
        });
    }, { threshold: 0.1, rootMargin: "0px 0px -30px 0px" });

    // 观察浏览卡片的容器
    const browseContainer = document.getElementById("browse-cards");
    if (browseContainer) {
        // MutationObserver 监听卡片渲染
        const mo = new MutationObserver(() => {
            browseContainer.querySelectorAll(".col-lg-4").forEach(col => {
                col.style.opacity = "0";
                observer.observe(col);
            });
        });
        mo.observe(browseContainer, { childList: true });
    }
}
