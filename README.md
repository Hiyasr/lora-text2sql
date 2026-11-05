# lora-text2sql

[![CI](https://github.com/Hiyasr/lora-text2sql/actions/workflows/ci.yml/badge.svg)](https://github.com/Hiyasr/lora-text2sql/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)

**Fine-tuning a small open model for text-to-SQL with LoRA, and testing honestly whether it was worth it.**
A 0.5B-1.5B parameter model is fine-tuned on question-to-SQL pairs, then compared against a trivial baseline,
the same base model zero-shot, and a prompted API model. All four are scored on database schemas the model
never saw during training, and serving throughput is measured.

> **Status:** training, evaluation and serving code, plus unit tests and CI, are complete. Training and
> evaluation runs are in progress; the results tables are filled in only from measured runs.

## Pipeline

```
sql-create-context ─► split BY SCHEMA (no leakage) ─► prompt + label masking (loss on the SQL only)
      └─► LoRA (r=16, all attention and MLP projections) ─► adapter ─► merge ─► vLLM / FastAPI ─► benchmark

evaluate.py:  trivial | base zero-shot | LoRA | prompted API  ─►  exact · AST · schema-valid · latency
```

| Stage | What it does |
|---|---|
| Data | loads `b-mc2/sql-create-context` and holds out whole schemas, so the test set contains no table layout seen in training |
| Training | LoRA on a small instruct model, loss computed only on the SQL tokens |
| Evaluation | the same held-out examples are scored for all four approaches |
| Serving | adapter merged into the base weights, served with vLLM or the bundled FastAPI endpoint |
| Benchmark | latency percentiles and throughput under concurrency |

## Quickstart

Offline checks (no GPU, no network):

```bash
pip install -e ".[dev]"
pytest -q
```

Full run on a free Colab or Kaggle T4 GPU (the default 0.5B model fits):

```bash
pip install -e ".[train,serve,api,dev]"

# 1. train: also writes the held-out set to results/eval_split.json
python scripts/train.py --model Qwen/Qwen2.5-0.5B-Instruct --n-train 8000 --out outputs/lora

# 2. evaluate all approaches on identical unseen-schema examples (--api needs ANTHROPIC_API_KEY)
python scripts/evaluate.py --adapter outputs/lora --api

# 3. optional: merge and serve (vLLM needs a GPU machine)
python scripts/merge.py --model Qwen/Qwen2.5-0.5B-Instruct --adapter outputs/lora --out outputs/merged
vllm serve outputs/merged --port 8001
python scripts/bench.py --url http://localhost:8001/v1/completions \
    --model outputs/merged --concurrency 1 8 32
```

If you hit a CUDA out-of-memory error, add `--batch 4 --accum 4` to the training command.

A lightweight API is also included:

```bash
MODEL=Qwen/Qwen2.5-0.5B-Instruct ADAPTER=outputs/lora uvicorn t2sql.serve:app
# POST /sql  {"question": "...", "schema_sql": "CREATE TABLE ..."}
```

## Metrics

| Metric | Definition |
|---|---|
| **exact match** | normalized SQL string equality (whitespace, case and quoting ignored) |
| **AST match** | equality of the parsed query trees (sqlglot), lenient to formatting |
| **schema-valid** | the query plans in SQLite against the `CREATE TABLE` statements in the context; catches hallucinated tables and columns and syntax errors |
| **latency** | p50 / p95 seconds per query |
| **throughput** | requests per second and output tokens per second under concurrency (`bench.py`) |

The dataset has no row data, so none of these is execution accuracy. A query can be schema-valid and still
return the wrong rows.

## Results

_Held-out set size, base model and API model will be listed here once the runs are complete._

### Accuracy and latency

| approach | exact match | AST match | schema-valid | p50 (s) | p95 (s) |
|---|---|---|---|---|---|
| trivial baseline (`SELECT *` from the first table) | TBD | TBD | TBD | TBD | TBD |
| base model, zero-shot | TBD | TBD | TBD | TBD | TBD |
| LoRA fine-tuned | TBD | TBD | TBD | TBD | TBD |
| prompted API model | TBD | TBD | TBD | TBD | TBD |

### Serving

| concurrency | requests/s | output tokens/s | p50 (s) | p95 (s) |
|---|---|---|---|---|
| 1 | TBD | TBD | TBD | TBD |
| 8 | TBD | TBD | TBD | TBD |
| 32 | TBD | TBD | TBD | TBD |

## Failure analysis

_To be written after the evaluation runs: a categorized breakdown of the fine-tuned model's errors
(hallucinated columns or tables, valid but different query, wrong aggregation or condition, formatting-only
differences), five concrete examples from `results/predictions.json`, and the cases where the API model wins._

## Design notes

- **One prompt builder** (`prompts.py`) is shared by training, evaluation and serving. A mismatch between them silently ruins results.
- **Held-out schemas, not held-out rows.** Splitting by schema stops the model from being rewarded for memorizing table layouts.
- **Identical test set for every approach.** The held-out set is saved at training time and reused by all evaluations.
- **fp32 weights with fp16 autocast** so training works on a T4, which has no bf16 support.

## Limitations

- Single dataset and a single training seed, so small differences between approaches may be noise.
- No execution accuracy, because the dataset ships without row data.
- Questions in this dataset are simpler than production analytics queries.
- The API comparison reflects one model and one prompt; a tuned prompt could change the gap.

## Project layout

```
t2sql/      prompts, data loading and schema-level split, SQL scoring, generation backends, API server
scripts/    train, evaluate, merge, bench
tests/      SQL normalization and validity, schema-leakage check, label masking, prompt format
results/    comparison table, predictions and the held-out split (model weights are git-ignored)
```

## CI

`.github/workflows/ci.yml` runs the offline unit tests on every push and pull request.

## Data

Training data comes from the `b-mc2/sql-create-context` dataset on Hugging Face. Check its license before redistributing any of it.
