"""
agent.py
--------
This is the AGENT LOOP box from your diagram, implemented as a class.

    1. Build prompt        -> build_messages()
    2. Call Claude/OpenAI  -> call_model()
    3. Parse response      -> parser.parse_agent_response()
    4. Execute tool        -> execute_tool()
    5. Append to history   -> done inline in run()
    6. Loop / Done         -> the for-loop in run()

No agent framework is used here -- just the OpenAI client, a while/for
loop, and plain Python dicts for history. This is intentional: the
point of Week 1 is to see the loop, not hide it behind a library.
"""

import os

from openai import OpenAI

from prompts import SYSTEM_PROMPT, AGENT_RESPONSE_FORMAT
from parser import parse_agent_response, is_final_answer, AgentResponseError
from tools import calculator, web_search, file_read, file_write


# Maps the "action" string the model picks -> the real Python function.
# Every tool function takes the action_input dict and returns a string
# observation (this is the contract every tool in tools.py follows).
TOOL_REGISTRY = {
    "calculator": lambda inp: calculator(inp.get("expression")),
    "web_search": lambda inp: web_search(inp.get("query")),
    "file_read": lambda inp: file_read(inp.get("filename")),
    "file_write": lambda inp: file_write(inp.get("filename"), inp.get("content")),
}

MAX_ITERATIONS = 8  # safety valve so a confused model can't loop forever


class ReActAgent:
    def __init__(self, model: str = None, verbose: bool = True):
        self.client = OpenAI()  # reads OPENAI_API_KEY from the environment
        self.model = model or os.getenv("AGENT_MODEL", "gpt-5.5")
        self.verbose = verbose

    # ---------- Step 1: Build prompt (history + available tools) ----------
    def build_messages(self, history: list) -> list:
        # SYSTEM_PROMPT already documents the available tools, so we
        # just prepend it to whatever conversation has happened so far.
        return [{"role": "system", "content": SYSTEM_PROMPT}] + history

    # ---------- Step 2: Call the model ----------
    def call_model(self, messages: list) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            response_format=AGENT_RESPONSE_FORMAT,
        )
        return response.choices[0].message.content

    # ---------- Step 4: Execute tool ----------
    def execute_tool(self, action: str, action_input: dict) -> str:
        tool_fn = TOOL_REGISTRY.get(action)
        if tool_fn is None:
            return f"Error: unknown tool '{action}'"
        try:
            return tool_fn(action_input or {})
        except Exception as error:
            return f"Tool execution error: {error}"

    def _log(self, *args):
        if self.verbose:
            print(*args)

    # ---------- The full loop: steps 1-6 ----------
    def run(self, user_query: str) -> str:
        history = [{"role": "user", "content": user_query}]

        for step in range(1, MAX_ITERATIONS + 1):
            self._log(f"\n--- Step {step} ---")

            # 1. Build prompt
            messages = self.build_messages(history)

            # 2. Call Claude/OpenAI API
            raw_content = self.call_model(messages)

            # 3. Parse response
            try:
                parsed = parse_agent_response(raw_content)
            except AgentResponseError as error:
                self._log("Parse error:", error)
                # feed the error back so the model can self-correct,
                # instead of crashing the whole agent
                history.append({"role": "assistant", "content": raw_content})
                history.append({
                    "role": "user",
                    "content": f"Observation: your last reply was invalid ({error}). "
                                f"Reply again using the required JSON format.",
                })
                continue

            self._log("Reasoning:", parsed["reasoning_summary"])
            self._log("Action:", parsed["action"], parsed.get("action_input"))

            # keep the model's own turn in history so it remembers its plan
            history.append({"role": "assistant", "content": raw_content})

            # ---- Tool Call? decision diamond ----
            if is_final_answer(parsed):
                return parsed["final_answer"]

            # 4. Execute tool
            observation = self.execute_tool(parsed["action"], parsed.get("action_input"))
            self._log("Observation:", observation)

            # 5. Append result to history (then loop back to step 1)
            history.append({"role": "user", "content": f"Observation: {observation}"})

        return "Stopped: reached max iterations without a final answer."
