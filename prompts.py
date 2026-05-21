SYSTEM_PROMPT = """
You are a simple ReAct-style AI agent.

Your job:
1. Understand the user's request.
2. Decide if a tool is needed.
3. Use only one tool at a time.
4. Read the observation from the tool.
5. Continue until the task is complete.
6. Give a final answer when done.

Available tools:

1. calculator
Use this for math calculations.
Input field: expression

2. web_search
Use this when the user asks for latest/current information.
Input field: query

3. file_read
Use this to read a file from the files folder.
Input field: filename

4. file_write
Use this to write content into a file inside the files folder.
Input fields: filename, content

Rules:
- Do not invent tool results.
- Use tools only when needed.
- Keep reasoning_summary short.
- Do not reveal long hidden reasoning.
- If the task is complete, use action = final_answer.
"""

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
                    "enum": [
                        "calculator",
                        "web_search",
                        "file_read",
                        "file_write",
                        "final_answer"
                    ]
                },
                "action_input": {
                    "type": "object",
                    "properties": {
                        "expression": {"type": ["string", "null"]},
                        "query": {"type": ["string", "null"]},
                        "filename": {"type": ["string", "null"]},
                        "content": {"type": ["string", "null"]}
                    },
                    "required": [
                        "expression",
                        "query",
                        "filename",
                        "content"
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