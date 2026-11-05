# t2sql — LoRA fine-tuning a small open model for text-to-SQL, with an honest comparison

Fine-tune a 0.5B–1.5B open model with LoRA on text-to-SQL, then compare it against (a) a trivial
baseline, (b) the same base model zero-shot, and (c) a prompted API model — on schemas the model
never saw in training — and measure serving throughput.

```
sql-create-context ─► split BY SCHEMA (no leakage) ─► prompt/label masking (loss on SQL only)
      └─► LoRA (r=16, all attn+MLP projections) ─► adapter ─► merge ─► vLLM / FastAPI ─► bench
evaluate.py: trivial | base zero-shot | LoRA | API  ──►  exact · AST · schema-valid · latency
```

## Run (Colab/Kaggle T4 is enough for the default 0.5B model)
```bash
pip install -e ".[train,serve,api,dev]"
python scripts/train.py --model Qwen/Qwen2.5-0.5B-Instruct --n-train 8000 --out outputs/lora
python scripts/evaluate.py --adapter outputs/lora --api        # --api needs ANTHROPIC_API_KEY
python scripts/merge.py --model Qwen/Qwen2.5-0.5B-Instruct --adapter outputs/lora --out outputs/merged
vllm serve outputs/merged --port 8001                           # needs a GPU machine
python scripts/bench.py --url http://localhost:8001/v1/completions --model outputs/merged --concurrency 1 8 32
```
Offline checks (no GPU/network): `pip install -e ".[dev]" && pytest -q`

## Metrics (`t2sql/sqlcheck.py`)
- **exact match** after normalization, **AST match** (sqlglot) — lenient to formatting.
- **schema-valid**: the query plans in SQLite against the `CREATE TABLE` context; catches hallucinated
  tables/columns. (There is no row data, so this is *not* execution accuracy — say so in your write-up.)
- **latency** p50/p95 per query, and **throughput** under concurrency via `bench.py`.

## Results (fill from YOUR run; `results/comparison.md` is written by evaluate.py)
| approach | exact | AST | schema-valid | p50 s | p95 s |
|---|---|---|---|---|---|
| trivial baseline | | | | | |
| base zero-shot | | | | | |
| LoRA fine-tuned | | | | | |
| prompted API model | | | | | |

Serving (from bench.py): | concurrency | req/s | output tok/s | p50 | p95 |

## Write-up checklist
Report where the API model wins (it may on hard schemas) and the cost/latency trade-off, show 5
failure examples from `results/predictions.json`, and state limitations: single dataset, single seed,
synthetic-style questions, no execution accuracy.

## Design notes
- Same prompt builder for train/eval/serve (`prompts.py`) — a mismatch silently wrecks results.
- fp32 weights + fp16 autocast so training works on T4 (no bf16).
- The held-out set is saved at train time (`results/eval_split.json`) so every approach is scored on identical examples.
