"""
parser.py
---------
Custom parser for the ReAct agent's structured JSON output.

The model is forced (via prompts.AGENT_RESPONSE_FORMAT, a strict JSON
schema passed as `response_format`) to always reply with an object shaped
like:

    {
        "reasoning_summary": "...",
        "action": "calculator" | "web_search" | "file_read" | "file_write"
                  | "human_handoff" | "final_answer",
        "action_input": {"expression": ..., "query": ..., "filename": ...,
                          "content": ..., "question": ...},
        "final_answer": "..." | null
    }

Even with structured outputs turned on, real-world LLM calls can still
fail in a few ways: the SDK can return None, the JSON can be malformed or
wrapped in markdown fences, required keys can be missing, or the model can
invent an action name that isn't in the tool registry. This module is the
single place that turns "raw model text" into a validated Python dict, or
raises a descriptive AgentResponseError that agent.py feeds back to the
model so it can self-correct on the next turn.
"""

import json

from tools import TOOL_REGISTRY

# Every action the model is allowed to pick: every registered tool, plus
# the two special control-flow actions handled directly inside agent.py.
VALID_ACTIONS = set(TOOL_REGISTRY.keys()) | {"human_handoff", "final_answer"}

REQUIRED_TOP_LEVEL_KEYS = {"reasoning_summary", "action", "action_input", "final_answer"}


class AgentResponseError(Exception):
    """Raised whenever the model's raw output can't be turned into a valid agent step."""


def _extract_json_block(raw_content: str) -> str:
    """
    Structured outputs should return pure JSON, but in practice models
    (and some providers) occasionally wrap it in ```json fences or add
    stray whitespace. This pulls out the {...} block defensively instead
    of trusting raw_content as-is.
    """
    if raw_content is None:
        raise AgentResponseError("Model returned an empty response (None).")

    text = raw_content.strip()
    if not text:
        raise AgentResponseError("Model returned an empty string.")

    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise AgentResponseError("No JSON object found in model response.")

    return text[start:end + 1]


def parse_agent_response(raw_content: str) -> dict:
    """
    Turn the model's raw string output into a validated dict.

    Raises AgentResponseError with a human-readable reason on any
    failure, so the caller (agent.py) can feed the error back to the
    model instead of crashing the whole agent loop.
    """
    json_block = _extract_json_block(raw_content)

    try:
        parsed = json.loads(json_block)
    except json.JSONDecodeError as error:
        raise AgentResponseError(f"Invalid JSON ({error.msg} at position {error.pos}).") from error

    if not isinstance(parsed, dict):
        raise AgentResponseError("Top-level JSON value must be an object.")

    missing = REQUIRED_TOP_LEVEL_KEYS - parsed.keys()
    if missing:
        raise AgentResponseError(f"Missing required field(s): {', '.join(sorted(missing))}.")

    action = parsed.get("action")
    if action not in VALID_ACTIONS:
        raise AgentResponseError(
            f"Unknown action '{action}'. Must be one of: {', '.join(sorted(VALID_ACTIONS))}."
        )

    action_input = parsed.get("action_input")
    if action_input is not None and not isinstance(action_input, dict):
        raise AgentResponseError("'action_input' must be an object (or null).")

    if action == "final_answer" and not parsed.get("final_answer"):
        raise AgentResponseError("action is 'final_answer' but 'final_answer' is empty/null.")

    reasoning = parsed.get("reasoning_summary")
    if not isinstance(reasoning, str):
        raise AgentResponseError("'reasoning_summary' must be a string.")

    # Normalize so downstream code never has to check for None.
    parsed["action_input"] = action_input or {}
    return parsed


def is_final_answer(parsed: dict) -> bool:
    return parsed.get("action") == "final_answer"


def is_human_handoff(parsed: dict) -> bool:
    return parsed.get("action") == "human_handoff"
