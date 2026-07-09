"""唐诗生成 API — 关键词 / 场景描述 双模式"""
import re, json, random
from flask import Flask, request, jsonify, render_template_string
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

app = Flask(__name__)
try:
    from flask_cors import CORS; CORS(app)
except ImportError: pass

# ============ 加载模型 ============
print("加载模型...")

def load_model(path):
    tok = AutoTokenizer.from_pretrained(path)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(path)
    return tok, mdl

# 尝试加载分开训练的模型，失败则用通用模型
MODELS = {}
for style, dirname in [("wuyan","bart-wuyan"), ("qiyan","bart-qiyan"), ("song","bart-song"), ("tang","bart-tang")]:
    try:
        t, m = load_model(f"models/{dirname}")
        MODELS[style] = (t, m)
        print(f"  {style}: models/{dirname}")
    except Exception as e:
        print(f"  {style}: 未加载 ({e})")

if not MODELS:
    print("[ERROR] 无可用模型! 请先训练")
    exit(1)

# 默认使用第一个可用模型
DEFAULT_STYLE = "tang" if "tang" in MODELS else list(MODELS.keys())[0]
tokenizer, model = MODELS[DEFAULT_STYLE]

# 扩写索引
try:
    with open("models/firstline_index.json", encoding="utf-8") as f:
        EXPAND_INDEX = json.load(f)
except Exception:
    EXPAND_INDEX = {}

print(f"就绪! 可用模型: {list(MODELS.keys())}, 扩写索引: {len(EXPAND_INDEX)} 词\n")

# ============ 扩写 ============
def expand_keyword(keyword, style):
    """关键词 → 真实诗歌首联"""
    if keyword in EXPAND_INDEX and len(EXPAND_INDEX[keyword]) >= 2:
        candidates = EXPAND_INDEX[keyword]
        if style == "wuyan":
            filtered = [l for l in candidates if 10 <= len(re.findall(r"[一-鿿]", l)) <= 12]
        elif style == "qiyan":
            filtered = [l for l in candidates if 14 <= len(re.findall(r"[一-鿿]", l)) <= 16]
        else:
            filtered = candidates
        if filtered:
            return random.choice(filtered)
    return keyword

# ============ 路由 ============
HTML = """
<!DOCTYPE html>
<html lang="zh">
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0"><title>诗词生成</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:"Microsoft YaHei",sans-serif;background:linear-gradient(135deg,#1a1a2e,#16213e);min-height:100vh;display:flex;justify-content:center;align-items:center;padding:20px}
.container{background:rgba(255,255,255,.95);border-radius:16px;padding:32px;max-width:640px;width:100%;box-shadow:0 20px 60px rgba(0,0,0,.3)}
h1{text-align:center;color:#1a1a2e;margin-bottom:4px;font-size:26px}
.sub{text-align:center;color:#888;margin-bottom:20px;font-size:13px}
.tabs{display:flex;gap:4px;margin-bottom:16px}
.tab{flex:1;padding:10px;text-align:center;background:#eee;border-radius:8px 8px 0 0;cursor:pointer;font-size:14px;transition:all .2s}
.tab.active{background:#667eea;color:#fff}
label{display:block;margin-bottom:6px;color:#333;font-weight:bold;font-size:14px}
input,textarea,select{width:100%;padding:10px;border:2px solid #ddd;border-radius:8px;font-size:15px;margin-bottom:12px;transition:border .3s;font-family:inherit}
textarea{height:100px;resize:vertical}
input:focus,textarea:focus,select:focus{outline:none;border-color:#667eea}
.row{display:flex;gap:12px}
.row>*{flex:1}
button{width:100%;padding:12px;background:linear-gradient(135deg,#667eea,#764ba2);color:#fff;border:none;border-radius:8px;font-size:17px;cursor:pointer;transition:transform .2s}
button:hover{transform:translateY(-2px)}
button:disabled{opacity:.6;cursor:not-allowed}
.result{margin-top:20px;padding:20px;background:#f8f9fa;border-radius:8px;border-left:4px solid #667eea;display:none}
.result.show{display:block}
.result .poem{font-size:20px;line-height:2.2;text-align:center;color:#333;white-space:pre-line}
.result .meta{margin-top:12px;font-size:12px;color:#999;text-align:center}
.error{color:#e74c3c;text-align:center;margin-top:12px;display:none}
.mode-hint{font-size:12px;color:#999;margin-top:-8px;margin-bottom:12px}
</style>
</head>
<body>
<div class="container">
<h1>诗词生成</h1>
<p class="sub">BART Encoder-Decoder · 唐诗</p>

<div class="tabs">
<div class="tab active" onclick="switchMode('keyword')">关键词模式</div>
<div class="tab" onclick="switchMode('scene')">场景描述模式</div>
</div>

<div id="mode-keyword">
<label>主题 / 关键词</label>
<input id="keyword" value="春风" placeholder="如: 春风、明月、边塞..." maxlength="20">
<p class="mode-hint">输入1-4个字的主题词，系统自动扩写为诗歌首联</p>
</div>

<div id="mode-scene" style="display:none">
<label>场景描述</label>
<textarea id="scene" placeholder="例如: 春天的早晨，微风吹过湖面，柳絮纷飞，远处有笛声传来"></textarea>
<p class="mode-hint">输入一段场景或情感描述，AI 将其写成诗</p>
</div>

<div class="row">
<select id="style">
<option value="wuyan">五言</option><option value="qiyan">七言</option><option value="song">宋词</option>
</select>
<select id="lines">
<option value="4">4 行</option><option value="6">6 行</option><option value="8">8 行</option>
</select>
</div>

<button onclick="generate()">生成诗歌</button>
<div class="error" id="error"></div>
<div class="result" id="result">
<div class="poem" id="poem"></div>
<div class="meta" id="meta"></div>
</div>
</div>

<script>
var currentMode = "keyword";
function switchMode(mode){
    currentMode = mode;
    document.querySelectorAll(".tab").forEach(function(t,i){
        t.classList.toggle("active", (i===0 && mode==="keyword") || (i===1 && mode==="scene"));
    });
    document.getElementById("mode-keyword").style.display = mode==="keyword" ? "block" : "none";
    document.getElementById("mode-scene").style.display = mode==="scene" ? "block" : "none";
}
function generate(){
    var payload = {
        style: document.getElementById("style").value,
        num_lines: parseInt(document.getElementById("lines").value),
        mode: currentMode
    };
    if(currentMode==="keyword"){
        var kw = document.getElementById("keyword").value.trim();
        if(!kw){showError("请输入关键词");return}
        payload.keyword = kw;
    } else {
        var sc = document.getElementById("scene").value.trim();
        if(!sc){showError("请输入场景描述");return}
        payload.scene = sc;
    }
    document.getElementById("error").style.display="none";
    document.getElementById("result").classList.remove("show");
    var btn=document.querySelector("button");btn.disabled=true;btn.textContent="生成中...";
    fetch("/api/generate",{
        method:"POST",headers:{"Content-Type":"application/json"},
        body:JSON.stringify(payload)
    }).then(function(r){return r.json()}).then(function(data){
        btn.disabled=false;btn.textContent="生成诗歌";
        if(data.success){
            document.getElementById("poem").textContent=data.poem;
            document.getElementById("meta").textContent="耗时: "+(data.time||"")+" | 模式: "+(data.mode||"")+" | 格式: "+(data.style||"");
            document.getElementById("result").classList.add("show");
        }else{showError(data.error||"生成失败");}
    }).catch(function(e){btn.disabled=false;btn.textContent="生成诗歌";showError("网络错误: "+e);});
}
function showError(msg){var e=document.getElementById("error");e.textContent=msg;e.style.display="block";}
document.getElementById("keyword").addEventListener("keydown",function(e){if(e.key==="Enter")generate();});
</script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/api/generate", methods=["POST"])
def api_generate():
    import time, torch
    data = request.get_json(silent=True) or {}
    mode = data.get("mode","keyword")
    style = data.get("style","wuyan")
    num_lines = int(data.get("num_lines",4))

    # 选择模型
    if style in MODELS:
        tok, mdl = MODELS[style]
    else:
        tok, mdl = tokenizer, model

    start = time.time()
    try:
        # 宋词用"写一首词"，唐诗用"写诗"
        if style == "song":
            action = "写一首词，"
            num_lines = num_lines * 3  # 词的行数更灵活
        else:
            action = "写诗："

        if mode == "scene":
            desc = data.get("scene","").strip()
            if not desc: return jsonify({"success":False,"error":"请输入场景描述"}), 400
            prompt = f"{action}{desc}。"
        else:
            kw = data.get("keyword","").strip()
            if not kw: return jsonify({"success":False,"error":"请输入关键词"}), 400
            prompt = f"{action}描写{kw}的意境。"

        torch.manual_seed(int(time.time() * 1000) % (2**31))
        inputs = tok(prompt, return_tensors="pt")
        gen_kwargs = {k:v for k,v in inputs.items() if k!="token_type_ids"}

        outputs = mdl.generate(**gen_kwargs, max_new_tokens=num_lines*16,
            temperature=0.85, do_sample=True, top_p=0.92, top_k=60,
            repetition_penalty=1.2, num_beams=1, no_repeat_ngram_size=3)

        poem = tok.decode(outputs[0], skip_special_tokens=True)
        lines = re.split(r"[，。！？、；：\n]+", poem)
        expected_len = 7 if style == "qiyan" else (5 if style == "wuyan" else 0)
        lines = [l.strip() for l in lines if l.strip() and len(re.findall(r"[一-鿿]",l)) >= (expected_len or 3)]

        # 严格检查: 行数+字数
        def is_valid():
            if len(lines) < num_lines:
                return False
            if style == "song":
                return True
            return all(len(re.findall(r"[一-鿿]", l)) == expected_len for l in lines[:num_lines])

        retries = 0
        while not is_valid() and retries < 3:
            retries += 1
            torch.manual_seed(int(time.time() * (1000 + retries)) % (2**31))
            outputs = mdl.generate(**gen_kwargs, max_new_tokens=num_lines*18,
                temperature=0.85, do_sample=True, top_p=0.92, top_k=60,
                repetition_penalty=1.2, num_beams=1, no_repeat_ngram_size=3)
            poem = tok.decode(outputs[0], skip_special_tokens=True)
            lines = re.split(r"[，。！？、；：\n]+", poem)
            lines = [l.strip() for l in lines if l.strip() and len(re.findall(r"[一-鿿]",l)) >= (expected_len or 3)]

        # 确保行数
        if len(lines) < num_lines:
            extra_input = tok(prompt + "，" + "，".join(lines[-2:]),
                              return_tensors="pt")
            extra_kwargs = {k:v for k,v in extra_input.items() if k!="token_type_ids"}
            extra_out = mdl.generate(**extra_kwargs, max_new_tokens=num_lines*16,
                temperature=0.85, do_sample=True, top_p=0.92, top_k=60,
                repetition_penalty=1.2, num_beams=1, no_repeat_ngram_size=3)
            extra_text = tok.decode(extra_out[0], skip_special_tokens=True)
            extra_lines = re.split(r"[，。！？、；：\n]+", extra_text)
            extra_lines = [l.strip() for l in extra_lines
                          if l.strip() and len(re.findall(r"[一-鿿]",l))>=3]
            existing = set(lines)
            for l in extra_lines:
                if l not in existing and len(lines) < num_lines+4:
                    lines.append(l); existing.add(l)

        poem_text = "\n".join(lines[:num_lines])
        elapsed = int((time.time()-start)*1000)
        return jsonify({"success":True, "poem":poem_text,
                        "time":f"{elapsed}ms", "mode":mode, "style":style})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"success":False,"error":str(e)}), 500

if __name__ == "__main__":
    print("="*50)
    print("  诗词生成服务: http://localhost:5000")
    print("="*50)
    app.run(host="0.0.0.0", port=5000, debug=False)
