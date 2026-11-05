"""One prompt format shared by training, evaluation and serving (mismatch here silently ruins results)."""

def build_prompt(question: str, context: str) -> str:
    return f"### Schema\n{context.strip()}\n### Question\n{question.strip()}\n### SQL\n"


def clean_generation(text: str) -> str:
    """Models should emit SQL then EOS; cut at the first blank line / next header as a safety net."""
    text = text.split("###")[0].strip()
    return text.split("\n\n")[0].strip()
