"""Merge the LoRA adapter into the base weights so any inference server (vLLM, TGI) can load it.
python scripts/merge.py --model Qwen/Qwen2.5-0.5B-Instruct --adapter outputs/lora-qwen0.5b --out outputs/merged"""
import argparse, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--adapter", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
m = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float16), a.adapter).merge_and_unload()
m.save_pretrained(a.out); AutoTokenizer.from_pretrained(a.model).save_pretrained(a.out)
print("merged model ->", a.out)
