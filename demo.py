"""唐诗生成演示 — 本地运行"""
import re, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL_DIR = "models/bart-tang"

print("加载模型中...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)
print("模型加载完成!\n")

# 保存到文件避免终端乱码
output_lines = []

tests = ["春风", "明月", "边塞", "离别", "江南", "相思", "秋日", "登高"]
for kw in tests:
    inputs = tokenizer(kw, return_tensors="pt")
    gen_kwargs = {k: v for k, v in inputs.items() if k != "token_type_ids"}
    outputs = model.generate(**gen_kwargs, max_new_tokens=56, temperature=0.8,
                             do_sample=True, top_p=0.9, top_k=50, repetition_penalty=1.2)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    lines = re.split(r"[，。！？、；：\n]+", text)
    lines = [l.strip() for l in lines if l.strip() and len(re.findall(r"[一-鿿]", l)) >= 4][:4]

    result = "【{}】".format(kw)
    output_lines.append(result)
    for l in lines:
        output_lines.append("  " + l)
    output_lines.append("")

with open("demo_output.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(output_lines))

print("\n".join(output_lines))
print("\n结果已保存到 demo_output.txt")
