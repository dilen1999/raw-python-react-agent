import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

with patch("openai.OpenAI"):
    from agent import ReActAgent


class TestAgentLoop(unittest.TestCase):
    def _make_agent(self, scripted_responses, input_fn=None):
        with patch("openai.OpenAI"):
            agent = ReActAgent(verbose=False, input_fn=input_fn or (lambda prompt: "test answer"))
        responses = iter(scripted_responses)
        agent.call_model = lambda messages: next(responses)
        return agent

    def test_immediate_final_answer(self):
        agent = self._make_agent([
            '{"reasoning_summary": "easy", "action": "final_answer", '
            '"action_input": {}, "final_answer": "42"}'
        ])
        self.assertEqual(agent.run("what is the answer"), "42")

    def test_tool_call_then_final_answer(self):
        agent = self._make_agent([
            '{"reasoning_summary": "need math", "action": "calculator", '
            '"action_input": {"expression": "3*3"}, "final_answer": null}',
            '{"reasoning_summary": "done", "action": "final_answer", '
            '"action_input": {}, "final_answer": "9"}',
        ])
        self.assertEqual(agent.run("what is 3*3"), "9")

    def test_malformed_response_recovers_on_retry(self):
        agent = self._make_agent([
            "not valid json",
            '{"reasoning_summary": "recovered", "action": "final_answer", '
            '"action_input": {}, "final_answer": "ok"}',
        ])
        self.assertEqual(agent.run("hello"), "ok")

    def test_human_handoff_flow(self):
        agent = self._make_agent(
            [
                '{"reasoning_summary": "unclear", "action": "human_handoff", '
                '"action_input": {"question": "Which city?"}, "final_answer": null}',
                '{"reasoning_summary": "got it", "action": "final_answer", '
                '"action_input": {}, "final_answer": "Colombo weather is sunny"}',
            ],
            input_fn=lambda prompt: "Colombo",
        )
        self.assertEqual(agent.run("what's the weather"), "Colombo weather is sunny")

    def test_max_iterations_reached(self):
        loop_response = (
            '{"reasoning_summary": "still thinking", "action": "calculator", '
            '"action_input": {"expression": "1+1"}, "final_answer": null}'
        )
        agent = self._make_agent([loop_response] * 10)
        result = agent.run("loop forever")
        self.assertTrue(result.startswith("Stopped: reached max iterations"))

    def test_repeated_malformed_responses_give_up(self):
        agent = self._make_agent(["not json"] * 5)
        result = agent.run("break it")
        self.assertTrue(result.startswith("Stopped: the model kept returning malformed"))

    def test_model_call_exception_fails_loudly(self):
        agent = self._make_agent([])

        def _boom(messages):
            raise ConnectionError("network down")

        agent.call_model = _boom
        result = agent.run("hi")
        self.assertTrue(result.startswith("Stopped: the model call failed"))


if __name__ == "__main__":
    unittest.main()
