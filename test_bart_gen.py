import torch, sys
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

m = AutoModelForSeq2SeqLM.from_pretrained("models/bart-wuyan/final").to("cuda")
t = AutoTokenizer.from_pretrained("models/bart-wuyan/final")

with open("bart_out.txt", "w", encoding="utf-8") as f:
    for kw in ["春", "月", "秋", "山", "夜", "花", "雪", "风", "江", "梦"]:
        for temp in [0.7, 0.9]:
            inp = t(kw, return_tensors="pt")
            inp.pop("token_type_ids", None)
            inp = {k: v.to("cuda") for k, v in inp.items()}
            out = m.generate(**inp, max_new_tokens=40, do_sample=True, temperature=temp, top_p=0.9)
            text = t.decode(out[0], skip_special_tokens=True)
            text = text.replace(" ", "")
            f.write("[" + kw + "] temp=" + str(temp) + "\n" + text + "\n\n")
print("Done")
