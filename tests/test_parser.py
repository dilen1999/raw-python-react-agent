import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from parser import parse_agent_response, is_final_answer, is_human_handoff, AgentResponseError


class TestParser(unittest.TestCase):
    def test_valid_final_answer(self):
        raw = ('{"reasoning_summary": "done", "action": "final_answer", '
               '"action_input": {}, "final_answer": "42"}')
        parsed = parse_agent_response(raw)
        self.assertTrue(is_final_answer(parsed))
        self.assertEqual(parsed["final_answer"], "42")

    def test_valid_tool_call(self):
        raw = ('{"reasoning_summary": "need math", "action": "calculator", '
               '"action_input": {"expression": "2+2"}, "final_answer": null}')
        parsed = parse_agent_response(raw)
        self.assertFalse(is_final_answer(parsed))
        self.assertEqual(parsed["action"], "calculator")
        self.assertEqual(parsed["action_input"]["expression"], "2+2")

    def test_human_handoff_detected(self):
        raw = ('{"reasoning_summary": "unclear", "action": "human_handoff", '
               '"action_input": {"question": "Which file?"}, "final_answer": null}')
        parsed = parse_agent_response(raw)
        self.assertTrue(is_human_handoff(parsed))
        self.assertEqual(parsed["action_input"]["question"], "Which file?")

    def test_strips_markdown_code_fences(self):
        raw = (
            '```json\n'
            '{"reasoning_summary": "ok", "action": "final_answer", '
            '"action_input": {}, "final_answer": "hi"}\n'
            '```'
        )
        parsed = parse_agent_response(raw)
        self.assertEqual(parsed["final_answer"], "hi")

    def test_null_action_input_becomes_empty_dict(self):
        raw = ('{"reasoning_summary": "ok", "action": "final_answer", '
               '"action_input": null, "final_answer": "hi"}')
        parsed = parse_agent_response(raw)
        self.assertEqual(parsed["action_input"], {})

    def test_none_response_raises(self):
        with self.assertRaises(AgentResponseError):
            parse_agent_response(None)

    def test_empty_string_raises(self):
        with self.assertRaises(AgentResponseError):
            parse_agent_response("   ")

    def test_invalid_json_raises(self):
        with self.assertRaises(AgentResponseError):
            parse_agent_response("this is not json at all")

    def test_missing_field_raises(self):
        raw = '{"reasoning_summary": "x", "action": "final_answer"}'
        with self.assertRaises(AgentResponseError):
            parse_agent_response(raw)

    def test_unknown_action_raises(self):
        raw = ('{"reasoning_summary": "x", "action": "delete_database", '
               '"action_input": {}, "final_answer": null}')
        with self.assertRaises(AgentResponseError):
            parse_agent_response(raw)

    def test_final_answer_action_without_text_raises(self):
        raw = ('{"reasoning_summary": "x", "action": "final_answer", '
               '"action_input": {}, "final_answer": null}')
        with self.assertRaises(AgentResponseError):
            parse_agent_response(raw)

    def test_non_object_action_input_raises(self):
        raw = ('{"reasoning_summary": "x", "action": "calculator", '
               '"action_input": "2+2", "final_answer": null}')
        with self.assertRaises(AgentResponseError):
            parse_agent_response(raw)


if __name__ == "__main__":
    unittest.main()
