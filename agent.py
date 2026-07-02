"""
agent.py
--------
This is the AGENT LOOP box from the architecture diagram, implemented as
a class.

    1. Build prompt          -> build_messages()
    2. Call the OpenAI API   -> call_model()
    3. Parse response        -> parser.parse_agent_response()
    4. Execute tool          -> execute_tool()
       (or pause for a human -> human_handoff())
    5. Append to history     -> done inline in run()
    6. Loop / Done           -> the for-loop in run()

No agent framework is used here -- just the OpenAI client, a for loop,
and plain Python dicts for history. That's intentional: the point of
this project is to see the loop, not hide it behind a library.
"""

import logging
import os

from openai import OpenAI

from prompts import SYSTEM_PROMPT, AGENT_RESPONSE_FORMAT
from parser import parse_agent_response, is_final_answer, is_human_handoff, AgentResponseError
from tools import TOOL_REGISTRY

logger = logging.getLogger("react_agent")

MAX_ITERATIONS = 8      # safety valve so a confused model can't loop forever
MAX_PARSE_RETRIES = 3   # safety valve for repeated malformed responses


class ReActAgent:
    def __init__(self, model: str = None, verbose: bool = True, input_fn=input):
        self.client = OpenAI()  # reads OPENAI_API_KEY from the environment
        self.model = model or os.getenv("AGENT_MODEL", "gpt-4o")
        self.verbose = verbose
        # input_fn is injectable so human_handoff doesn't have to be driven
        # by the real terminal -- tests pass a scripted callable instead.
        self.input_fn = input_fn

    # ---------- Step 1: Build prompt (history + available tools) ----------
    def build_messages(self, history: list) -> list:
        # SYSTEM_PROMPT already documents the available tools, so we just
        # prepend it to whatever conversation has happened so far.
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
        tool = TOOL_REGISTRY.get(action)
        if tool is None:
            return f"Error: unknown tool '{action}'"
        try:
            return tool["function"](action_input or {})
        except Exception as error:
            return f"Tool execution error: {error}"

    # ---------- Human handoff: pause the loop and ask a real person ----------
    def human_handoff(self, question: str) -> str:
        self._log(f"\n[Agent needs input] {question}")
        answer = self.input_fn(f"[Agent asks] {question}\nYou: ")
        answer = (answer or "").strip()
        return answer or "(no answer provided)"

    def _log(self, *args):
        logger.info(" ".join(str(a) for a in args))
        if self.verbose:
            print(*args)

    # ---------- The full loop: steps 1-6 ----------
    def run(self, user_query: str) -> str:
        history = [{"role": "user", "content": user_query}]
        parse_failures = 0

        for step in range(1, MAX_ITERATIONS + 1):
            self._log(f"\n--- Step {step} ---")

            # 1. Build prompt
            messages = self.build_messages(history)

            # 2. Call the model. API/network errors are not something the
            # model can "self-correct" on the next turn, so fail loudly
            # instead of burning iterations on a dead connection.
            try:
                raw_content = self.call_model(messages)
            except Exception as error:
                logger.exception("Model call failed")
                return f"Stopped: the model call failed ({error})."

            # 3. Parse response
            try:
                parsed = parse_agent_response(raw_content)
                parse_failures = 0
            except AgentResponseError as error:
                parse_failures += 1
                self._log("Parse error:", error)
                # feed the error back so the model can self-correct,
                # instead of crashing the whole agent
                history.append({"role": "assistant", "content": raw_content or ""})
                if parse_failures >= MAX_PARSE_RETRIES:
                    return (
                        "Stopped: the model kept returning malformed responses "
                        f"({parse_failures} attempts). Last error: {error}"
                    )
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

            if is_human_handoff(parsed):
                question = (parsed.get("action_input") or {}).get("question") \
                    or "Could you clarify what you'd like me to do?"
                answer = self.human_handoff(question)
                history.append({"role": "user", "content": f"Observation (human): {answer}"})
                continue

            # 4. Execute tool
            observation = self.execute_tool(parsed["action"], parsed.get("action_input"))
            self._log("Observation:", observation)

            # 5. Append result to history (then loop back to step 1)
            history.append({"role": "user", "content": f"Observation: {observation}"})

        return "Stopped: reached max iterations without a final answer."
