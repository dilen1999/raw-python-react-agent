"""
parser.py
---------
Maps to Step 3 in the flow diagram: "Parse response".

Because prompts.py forces the model to reply using OpenAI's strict
JSON-schema structured output, the raw content should already be valid
JSON. This file is still kept separate (rather than inlined into
agent.py) so that:

  1. Parsing/validation logic is testable on its own.
  2. If you ever swap providers (e.g. a model that doesn't support
     strict JSON schema), you only need to change this file.
"""

import json


class AgentResponseError(Exception):
    """Raised when the model's reply can't be parsed into a valid step."""


REQUIRED_KEYS = {"reasoning_summary", "action", "action_input", "final_answer"}

VALID_ACTIONS = {"calculator", "web_search", "file_read", "file_write", "final_answer"}


def parse_agent_response(raw_content: str) -> dict:
    """Turn the model's raw text into a dict, validating its shape.

    Raises AgentResponseError on anything malformed so agent.py can
    decide how to recover (e.g. feed the error back to the model).
    """
    if not raw_content:
        raise AgentResponseError("Model returned an empty response.")

    try:
        data = json.loads(raw_content)
    except json.JSONDecodeError as error:
        raise AgentResponseError(f"Model did not return valid JSON: {error}")

    if not isinstance(data, dict):
        raise AgentResponseError("Model response was valid JSON but not an object.")

    missing = REQUIRED_KEYS - data.keys()
    if missing:
        raise AgentResponseError(f"Response is missing required keys: {missing}")

    if data["action"] not in VALID_ACTIONS:
        raise AgentResponseError(f"Unknown action: {data['action']}")

    if data["action"] == "final_answer" and not data.get("final_answer"):
        raise AgentResponseError("action is final_answer but final_answer is empty.")

    return data


def is_final_answer(parsed: dict) -> bool:
    """The 'Tool Call?' decision diamond in the diagram."""
    return parsed.get("action") == "final_answer"
