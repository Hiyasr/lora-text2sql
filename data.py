"""Data loading with a leakage-safe split: held-out examples use schemas never seen in training."""
from __future__ import annotations
import random
from .prompts import build_prompt

DATASET = "b-mc2/sql-create-context"   # columns: question, context, answer  (check its license before redistributing)


def split_by_context(rows: list[dict], eval_frac: float = 0.05, seed: int = 0) -> tuple[list[dict], list[dict]]:
    ctxs = sorted({r["context"] for r in rows})
    rng = random.Random(seed)
    rng.shuffle(ctxs)
    held = set(ctxs[: max(1, int(len(ctxs) * eval_frac))])
    train = [r for r in rows if r["context"] not in held]
    ev = [r for r in rows if r["context"] in held]
    return train, ev


def load_splits(n_train: int = 8000, n_eval: int = 300, seed: int = 0):
    from datasets import load_dataset  # lazy: keeps unit tests light
    rows = [dict(r) for r in load_dataset(DATASET, split="train")]
    train, ev = split_by_context(rows, 0.05, seed)
    rng = random.Random(seed)
    rng.shuffle(train); rng.shuffle(ev)
    return train[:n_train], ev[:n_eval]


def encode_example(tok, row: dict, max_len: int = 512) -> dict:
    """Tokenize prompt+answer(+EOS); labels are -100 on the prompt so loss is only on the SQL."""
    p = tok(build_prompt(row["question"], row["context"]), add_special_tokens=False)["input_ids"]
    a = tok(row["answer"].strip(), add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    ids = (p + a)[:max_len]
    labels = ([-100] * len(p) + a)[:max_len]
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}
