"""
tools.py
--------
Every tool the agent can call lives here, plus a small self-registering
TOOL_REGISTRY so agent.py and prompts.py never need a hardcoded mapping
or hand-written tool list that can drift out of sync with the code.

To add a new tool:
    1. Write a function `def my_tool(action_input: dict) -> str: ...`
       that takes the action_input dict and returns a string observation.
    2. Decorate it with @register_tool("name", "one-line description").

That's it. agent.py picks it up via TOOL_REGISTRY, and prompts.py picks
it up via describe_tools() when it builds the system prompt.
"""

import ast
import operator
import os
from pathlib import Path

from openai import OpenAI

BASE_DIR = Path(__file__).resolve().parent
FILES_DIR = BASE_DIR / "files"
FILES_DIR.mkdir(exist_ok=True)

# -----------------------------
# Modular tool registry
# -----------------------------
TOOL_REGISTRY = {}


def register_tool(name: str, description: str):
    """Decorator that registers a tool function under `name`, with a
    human-readable `description` used to auto-build the system prompt."""

    def decorator(fn):
        TOOL_REGISTRY[name] = {"function": fn, "description": description}
        return fn

    return decorator


def describe_tools() -> str:
    """Render the registry as a numbered list for the system prompt, so
    prompts.py and tools.py can never drift out of sync."""
    lines = []
    for i, (name, meta) in enumerate(TOOL_REGISTRY.items(), start=1):
        lines.append(f"{i}. {name}\n   {meta['description']}")
    return "\n".join(lines)


# -----------------------------
# Safe calculator (AST-based, no eval())
# -----------------------------
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


@register_tool("calculator", "Evaluate a math expression. Input field: expression")
def calculator(action_input: dict) -> str:
    expression = (action_input or {}).get("expression")
    try:
        if not expression or len(expression) > 200:
            return "Error: Expression is empty or too long."
        tree = ast.parse(expression, mode="eval")
        result = _eval_math_node(tree.body)
        return str(result)
    except Exception as error:
        return f"Calculator error: {error}"


# -----------------------------
# File tools (sandboxed to ./files)
# -----------------------------
def _safe_file_path(filename: str) -> Path:
    if not filename:
        raise ValueError("Filename is required.")
    path = (FILES_DIR / filename).resolve()
    if not str(path).startswith(str(FILES_DIR.resolve())):
        raise ValueError("Access outside the files folder is not allowed.")
    return path


@register_tool("file_read", "Read a file from the files folder. Input field: filename")
def file_read(action_input: dict) -> str:
    filename = (action_input or {}).get("filename")
    try:
        path = _safe_file_path(filename)
        if not path.exists():
            return f"File not found: {filename}"
        return path.read_text(encoding="utf-8")
    except Exception as error:
        return f"File read error: {error}"


@register_tool(
    "file_write",
    "Write content into a file inside the files folder. Input fields: filename, content",
)
def file_write(action_input: dict) -> str:
    action_input = action_input or {}
    filename = action_input.get("filename")
    content = action_input.get("content")
    try:
        path = _safe_file_path(filename)
        path.write_text(content or "", encoding="utf-8")
        return f"File written successfully: {filename}"
    except Exception as error:
        return f"File write error: {error}"


# -----------------------------
# Web search tool
# -----------------------------
@register_tool("web_search", "Search the web for current/latest information. Input field: query")
def web_search(action_input: dict) -> str:
    query = (action_input or {}).get("query")
    try:
        if not query:
            return "Error: Search query is required."
        client = OpenAI()
        model = os.getenv("AGENT_MODEL", "gpt-4o")
        response = client.responses.create(
            model=model,
            tools=[{"type": "web_search"}],
            input=f"Search the web and give a short factual summary for this query: {query}",
        )
        return response.output_text
    except Exception as error:
        return f"Web search error: {error}"
