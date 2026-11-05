"""SQL scoring that needs no database contents: normalized exact match, AST match (sqlglot),
and schema-validity (does the query plan against the CREATE TABLE statements in the context?)."""
from __future__ import annotations
import re
import sqlite3


def normalize_sql(sql: str) -> str:
    s = sql.strip().rstrip(";").strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\s*([(),=<>])\s*", r"\1", s)
    return s.replace('"', "'")


def exact_match(pred: str, gold: str) -> bool:
    return normalize_sql(pred) == normalize_sql(gold)


def ast_match(pred: str, gold: str) -> bool:
    try:
        import sqlglot
        return sqlglot.parse_one(pred).sql(normalize=True) == sqlglot.parse_one(gold).sql(normalize=True)
    except Exception:
        return exact_match(pred, gold)


def is_valid(sql: str, context: str) -> bool:
    """True if SQLite can plan `sql` against the schema. Catches hallucinated tables/columns and syntax errors."""
    db = sqlite3.connect(":memory:")
    try:
        for stmt in context.split(";"):
            if stmt.strip():
                db.execute(stmt)
        db.execute("EXPLAIN " + sql.strip().rstrip(";"))
        return True
    except Exception:
        return False
    finally:
        db.close()


def score(pred: str, gold: str, context: str) -> dict:
    return {"exact": exact_match(pred, gold), "ast": ast_match(pred, gold), "valid": is_valid(pred, context)}
