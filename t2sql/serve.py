"""FastAPI endpoint. MODEL=Qwen/Qwen2.5-0.5B-Instruct ADAPTER=outputs/lora uvicorn t2sql.serve:app
Body: {"question": "...", "schema": "CREATE TABLE ..."}"""
import os
import time
from fastapi import FastAPI
from pydantic import BaseModel
from .generate import hf_backend
from .sqlcheck import is_valid

app = FastAPI(title="t2sql")
_gen = hf_backend(os.environ["MODEL"], os.environ.get("ADAPTER"))


class Req(BaseModel):
    question: str
    schema_sql: str


@app.post("/sql")
def sql(req: Req):
    t = time.perf_counter()
    q = _gen({"question": req.question, "context": req.schema_sql})
    return {"sql": q, "schema_valid": is_valid(q, req.schema_sql), "latency_s": round(time.perf_counter() - t, 3)}
