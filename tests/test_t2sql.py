from t2sql.data import split_by_context, encode_example
from t2sql.prompts import build_prompt, clean_generation
from t2sql.sqlcheck import normalize_sql, exact_match, ast_match, is_valid, score
from t2sql.generate import trivial_backend

CTX = "CREATE TABLE head (age INTEGER, name TEXT)"


class FakeTok:
    """One token per whitespace-separated word; id 0 is reserved for EOS."""
    eos_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [len(w) + 1 for w in text.split()]}


def test_normalize_and_exact():
    assert exact_match("SELECT  COUNT(*)  FROM head ;", "select count(*) from head")
    assert not exact_match("SELECT name FROM head", "SELECT age FROM head")
    assert normalize_sql('SELECT "a" ,b FROM t') == "select 'a',b from t"


def test_ast_match_is_more_lenient_than_exact():
    assert ast_match("SELECT name FROM head WHERE age > 3", "select name from head where age>3")


def test_validity_catches_hallucinated_columns_and_tables():
    assert is_valid("SELECT name FROM head WHERE age > 3", CTX)
    assert not is_valid("SELECT salary FROM head", CTX)
    assert not is_valid("SELECT name FROM employees", CTX)
    assert not is_valid("SELEC name FROM head", CTX)


def test_score_dict():
    s = score("SELECT name FROM head", "SELECT name FROM head", CTX)
    assert s == {"exact": True, "ast": True, "valid": True}


def test_split_has_no_schema_leakage():
    rows = [{"question": "q", "context": f"CREATE TABLE t{i % 20} (a INT)", "answer": "x"} for i in range(200)]
    train, ev = split_by_context(rows, 0.1, seed=1)
    assert ev and train
    assert not ({r["context"] for r in train} & {r["context"] for r in ev})


def test_labels_mask_prompt_and_end_with_eos():
    ex = encode_example(FakeTok(), {"question": "how many", "context": CTX, "answer": "SELECT COUNT(*) FROM head"})
    n_prompt = sum(1 for l in ex["labels"] if l == -100)
    assert 0 < n_prompt < len(ex["labels"])
    assert ex["labels"][-1] == 0 and ex["input_ids"][-1] == 0
    assert all(l == -100 for l in ex["labels"][:n_prompt]) and len(ex["labels"]) == len(ex["input_ids"])


def test_prompt_format_and_cleanup():
    p = build_prompt("How many?", CTX)
    assert p.endswith("### SQL\n") and CTX in p
    assert clean_generation("SELECT 1\n\n### Question\nfoo") == "SELECT 1"


def test_trivial_baseline_is_valid_but_wrong():
    row = {"context": CTX, "question": "names?", "answer": "SELECT name FROM head"}
    pred = trivial_backend()(row)
    assert is_valid(pred, CTX) and not exact_match(pred, row["answer"])
