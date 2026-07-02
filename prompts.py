"""
prompts.py
----------
The system prompt and the strict JSON schema (structured outputs) the
model must follow on every turn.

The tool list inside SYSTEM_PROMPT and the "action" enum inside
AGENT_RESPONSE_FORMAT are both generated straight from tools.TOOL_REGISTRY,
so the prompt can never go stale as tools are added, removed, or renamed.
"""

from tools import describe_tools, TOOL_REGISTRY

# Every action the model may pick: every registered tool, plus the two
# special control-flow actions the agent loop handles directly.
_ACTION_ENUM = list(TOOL_REGISTRY.keys()) + ["human_handoff", "final_answer"]

SYSTEM_PROMPT = f"""
You are a simple ReAct-style AI agent (Reasoning + Acting).

Your job:
1. Understand the user's request.
2. Decide if a tool is needed.
3. Use only one tool at a time.
4. Read the observation from the tool.
5. Continue until the task is complete.
6. Give a final answer when done.

Available tools:
{describe_tools()}

Special actions:
- human_handoff
  Use this ONLY when you are missing information only the user can supply
  (an ambiguous request, a decision only they can make, or you are stuck
  after a tool keeps failing). Input field: question
- final_answer
  Use this when the task is complete. Put the full answer for the user in
  the top-level "final_answer" field.

Rules:
- Do not invent tool results.
- Use tools only when needed.
- Keep reasoning_summary short (it is shown to the user, not hidden chain-of-thought).
- If you are unsure and truly need clarification, use human_handoff instead of guessing.
- If the task is complete, use action = final_answer.
""".strip()

AGENT_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "agent_step",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "reasoning_summary": {
                    "type": "string"
                },
                "action": {
                    "type": "string",
                    "enum": _ACTION_ENUM
                },
                "action_input": {
                    "type": "object",
                    "properties": {
                        "expression": {"type": ["string", "null"]},
                        "query": {"type": ["string", "null"]},
                        "filename": {"type": ["string", "null"]},
                        "content": {"type": ["string", "null"]},
                        "question": {"type": ["string", "null"]}
                    },
                    "required": [
                        "expression",
                        "query",
                        "filename",
                        "content",
                        "question"
                    ],
                    "additionalProperties": False
                },
                "final_answer": {
                    "type": ["string", "null"]
                }
            },
            "required": [
                "reasoning_summary",
                "action",
                "action_input",
                "final_answer"
            ],
            "additionalProperties": False
        }
    }
}
