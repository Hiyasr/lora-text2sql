"""Compare approaches on the SAME held-out (unseen-schema) set; writes results/comparison.md.
python scripts/evaluate.py --split results/eval_split.json --model Qwen/Qwen2.5-0.5B-Instruct \
       --adapter outputs/lora-qwen0.5b --api          # --api needs ANTHROPIC_API_KEY"""
import argparse, json, statistics, time
from t2sql.generate import trivial_backend, api_backend, hf_backend
from t2sql.sqlcheck import score

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="results/eval_split.json")
ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
ap.add_argument("--adapter", default=None)
ap.add_argument("--api", action="store_true")
ap.add_argument("--skip-base", action="store_true")
ap.add_argument("--n", type=int, default=None)
a = ap.parse_args()
ev = json.load(open(a.split))[: a.n]

backends = {"trivial (SELECT * first table)": trivial_backend()}
if not a.skip_base:
    backends[f"base: {a.model} (zero-shot)"] = hf_backend(a.model)
if a.adapter:
    backends[f"LoRA: {a.model}"] = hf_backend(a.model, a.adapter)
if a.api:
    backends["prompted API model"] = api_backend()

rows, detail = [], {}
for name, gen in backends.items():
    lat, s = [], []
    for r in ev:
        t = time.perf_counter(); pred = gen(r); lat.append(time.perf_counter() - t)
        s.append({**score(pred, r["answer"], r["context"]), "pred": pred, "gold": r["answer"]})
    detail[name] = s
    n = len(s)
    lat_sorted = sorted(lat)
    rows.append((name, sum(x["exact"] for x in s) / n, sum(x["ast"] for x in s) / n, sum(x["valid"] for x in s) / n,
                 statistics.median(lat), lat_sorted[min(n - 1, int(0.95 * (n - 1)))]))

md = ["| approach | exact match | AST match | schema-valid | p50 latency (s) | p95 latency (s) |", "|---|---|---|---|---|---|"]
md += [f"| {n} | {e:.3f} | {ast:.3f} | {v:.3f} | {p50:.3f} | {p95:.3f} |" for n, e, ast, v, p50, p95 in rows]
open("results/comparison.md", "w").write(f"Held-out examples: {len(ev)} (schemas unseen in training)\n\n" + "\n".join(md) + "\n")
json.dump(detail, open("results/predictions.json", "w"), indent=1)
print("\n".join(md))
