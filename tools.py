import ast
import operator
import os
from pathlib import Path
from openai import OpenAI


BASE_DIR = Path(__file__).resolve().parent
FILES_DIR = BASE_DIR / "files"
FILES_DIR.mkdir(exist_ok=True)


# -----------------------------
# Safe calculator
# -----------------------------

# Define allowed operators for safe evaluation
ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}

ALLOWED_UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _eval_math_node(node):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Only numbers are allowed.")

    if isinstance(node, ast.BinOp):
        operator_type = type(node.op)
        if operator_type not in ALLOWED_OPERATORS:
            raise ValueError("This math operator is not allowed.")
        left = _eval_math_node(node.left)
        right = _eval_math_node(node.right)
        return ALLOWED_OPERATORS[operator_type](left, right)

    if isinstance(node, ast.UnaryOp):
        operator_type = type(node.op)
        if operator_type not in ALLOWED_UNARY_OPERATORS:
            raise ValueError("This unary operator is not allowed.")
        value = _eval_math_node(node.operand)
        return ALLOWED_UNARY_OPERATORS[operator_type](value)

    raise ValueError("Invalid math expression.")


def calculator(expression: str) -> str:
    try:
        if not expression or len(expression) > 200:
            return "Error: Expression is empty or too long."

        tree = ast.parse(expression, mode="eval")
        result = _eval_math_node(tree.body)
        return str(result)

    except Exception as error:
        return f"Calculator error: {error}"


# -----------------------------
# File tools
# -----------------------------

def _safe_file_path(filename: str) -> Path:
    if not filename:
        raise ValueError("Filename is required.")

    path = (FILES_DIR / filename).resolve()

    if not str(path).startswith(str(FILES_DIR.resolve())):
        raise ValueError("Access outside files folder is not allowed.")

    return path


def file_read(filename: str) -> str:
    try:
        path = _safe_file_path(filename)

        if not path.exists():
            return f"File not found: {filename}"

        return path.read_text(encoding="utf-8")

    except Exception as error:
        return f"File read error: {error}"


def file_write(filename: str, content: str) -> str:
    try:
        path = _safe_file_path(filename)
        path.write_text(content or "", encoding="utf-8")
        return f"File written successfully: {filename}"

    except Exception as error:
        return f"File write error: {error}"


# -----------------------------
# Web search tool
# -----------------------------

def web_search(query: str) -> str:
    try:
        if not query:
            return "Error: Search query is required."

        client = OpenAI()
        model = os.getenv("AGENT_MODEL", "gpt-5.5")

        response = client.responses.create(
            model=model,
            tools=[{"type": "web_search"}],
            input=f"Search the web and give a short factual summary for this query: {query}"
        )

        return response.output_text

    except Exception as error:
        return f"Web search error: {error}"