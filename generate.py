"""Generation backends sharing one signature: gen(row) -> sql string."""
from __future__ import annotations
import os
import re
import time
from .prompts import build_prompt, clean_generation


def trivial_backend():
    """Naive baseline: SELECT * from the first table in the schema."""
    def gen(row):
        m = re.search(r"create table\s+(\w+)", row["context"], re.I)
        return f"SELECT * FROM {m.group(1)}" if m else "SELECT 1"
    return gen


def api_backend(model: str | None = None):
    import anthropic
    client, model = anthropic.Anthropic(), model or os.environ.get("T2SQL_API_MODEL", "claude-haiku-5-5")
    sys = "Translate the question into one SQLite query for the given schema. Output ONLY the SQL, no explanation."

    def gen(row):
        for attempt in range(4):
            try:
                r = client.messages.create(model=model, max_tokens=200, system=sys, messages=[
                    {"role": "user", "content": f"Schema:\n{row['context']}\n\nQuestion: {row['question']}"}])
                return clean_generation(re.sub(r"^```(?:sql)?|```$", "", "".join(b.text for b in r.content if b.type == "text").strip()).strip())
            except Exception:
                time.sleep(2 ** attempt)
        return ""
    gen.name = model
    return gen


def hf_backend(model_name: str, adapter: str | None = None, max_new_tokens: int = 128):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16 if dev == "cuda" else torch.float32)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.to(dev).eval()

    @torch.no_grad()
    def gen(row):
        ids = tok(build_prompt(row["question"], row["context"]), return_tensors="pt", add_special_tokens=False).to(dev)
        out = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
        return clean_generation(tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True))
    gen.name = model_name + (f"+{adapter}" if adapter else "")
    return gen
