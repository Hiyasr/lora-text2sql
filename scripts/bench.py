"""Serving benchmark: latency percentiles + throughput under concurrency.
HTTP mode targets any OpenAI-compatible /v1/completions server (e.g. vLLM on the merged model):
  vllm serve outputs/merged --port 8001
  python scripts/bench.py --url http://localhost:8001/v1/completions --model outputs/merged --concurrency 1 8 32
"""
import argparse, json, statistics, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from t2sql.prompts import build_prompt

ap = argparse.ArgumentParser()
ap.add_argument("--url", required=True); ap.add_argument("--model", required=True)
ap.add_argument("--split", default="results/eval_split.json")
ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 8])
ap.add_argument("--n", type=int, default=64)
a = ap.parse_args()
rows = json.load(open(a.split))[: a.n]


def call(row):
    body = json.dumps({"model": a.model, "prompt": build_prompt(row["question"], row["context"]),
                       "max_tokens": 128, "temperature": 0}).encode()
    t = time.perf_counter()
    out = json.load(urllib.request.urlopen(urllib.request.Request(a.url, body, {"Content-Type": "application/json"})))
    return time.perf_counter() - t, out.get("usage", {}).get("completion_tokens", 0)


print("| concurrency | req/s | output tok/s | p50 s | p95 s |\n|---|---|---|---|---|")
for c in a.concurrency:
    t0 = time.perf_counter()
    with ThreadPoolExecutor(c) as ex:
        res = list(ex.map(call, rows))
    wall = time.perf_counter() - t0
    lat = sorted(r[0] for r in res)
    print(f"| {c} | {len(rows) / wall:.2f} | {sum(r[1] for r in res) / wall:.1f} | "
          f"{statistics.median(lat):.3f} | {lat[min(len(lat) - 1, int(.95 * (len(lat) - 1)))]:.3f} |")
