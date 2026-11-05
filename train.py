"""LoRA fine-tune. Colab/Kaggle T4 friendly with the default 0.5B model.
python scripts/train.py --model Qwen/Qwen2.5-0.5B-Instruct --n-train 8000 --out outputs/lora-qwen0.5b"""
import argparse, json, torch
from transformers import (AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq,
                          Trainer, TrainingArguments)
from peft import LoraConfig, get_peft_model
from t2sql.data import load_splits, encode_example

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
ap.add_argument("--out", default="outputs/lora")
ap.add_argument("--n-train", type=int, default=8000)
ap.add_argument("--n-eval", type=int, default=300)
ap.add_argument("--epochs", type=float, default=1.0)
ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--accum", type=int, default=2)
ap.add_argument("--rank", type=int, default=16)
ap.add_argument("--report-to", default="none", help="none | wandb")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

tok = AutoTokenizer.from_pretrained(a.model)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
train, ev = load_splits(a.n_train, a.n_eval, a.seed)
json.dump(ev, open("results/eval_split.json", "w"))          # exact held-out set reused by scripts/evaluate.py
ds = [encode_example(tok, r) for r in train]

# fp32 weights + fp16 autocast: works on T4 (no bf16) without fp16-gradient errors
model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32)
model = get_peft_model(model, LoraConfig(
    r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
model.print_trainable_parameters()

args = TrainingArguments(
    output_dir=a.out, num_train_epochs=a.epochs, learning_rate=a.lr, lr_scheduler_type="cosine",
    warmup_ratio=0.03, per_device_train_batch_size=a.batch, gradient_accumulation_steps=a.accum,
    fp16=torch.cuda.is_available(), logging_steps=20, save_strategy="no", report_to=a.report_to,
    seed=a.seed, remove_unused_columns=False, group_by_length=True)
Trainer(model=model, args=args, train_dataset=ds,
        data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100)).train()
model.save_pretrained(a.out); tok.save_pretrained(a.out)
print("saved adapter to", a.out)
