import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools import calculator, file_read, file_write, TOOL_REGISTRY, FILES_DIR


class TestCalculator(unittest.TestCase):
    def test_basic_arithmetic(self):
        self.assertEqual(calculator({"expression": "2 + 2"}), "4")

    def test_operator_precedence(self):
        self.assertEqual(calculator({"expression": "2 + 3 * 4"}), "14")

    def test_power_and_mod(self):
        self.assertEqual(calculator({"expression": "2 ** 5"}), "32")
        self.assertEqual(calculator({"expression": "10 % 3"}), "1")

    def test_rejects_unsafe_expression(self):
        result = calculator({"expression": "__import__('os').system('ls')"})
        self.assertTrue(result.startswith("Calculator error"))

    def test_rejects_names(self):
        result = calculator({"expression": "os.getcwd()"})
        self.assertTrue(result.startswith("Calculator error"))

    def test_empty_expression(self):
        self.assertIn("Error", calculator({"expression": ""}))

    def test_missing_action_input(self):
        self.assertIn("Error", calculator({}))


class TestFileTools(unittest.TestCase):
    def setUp(self):
        self.test_file = "unit_test_tmp.txt"

    def tearDown(self):
        path = FILES_DIR / self.test_file
        if path.exists():
            path.unlink()

    def test_write_then_read(self):
        write_result = file_write({"filename": self.test_file, "content": "hello world"})
        self.assertIn("written successfully", write_result)
        read_result = file_read({"filename": self.test_file})
        self.assertEqual(read_result, "hello world")

    def test_read_missing_file(self):
        result = file_read({"filename": "does_not_exist.txt"})
        self.assertIn("File not found", result)

    def test_path_traversal_blocked(self):
        result = file_read({"filename": "../../etc/passwd"})
        self.assertIn("error", result.lower())

    def test_write_requires_filename(self):
        result = file_write({"filename": "", "content": "x"})
        self.assertIn("error", result.lower())


class TestRegistry(unittest.TestCase):
    def test_all_expected_tools_registered(self):
        for name in ("calculator", "web_search", "file_read", "file_write"):
            self.assertIn(name, TOOL_REGISTRY)

    def test_each_tool_has_description(self):
        for name, meta in TOOL_REGISTRY.items():
            self.assertTrue(meta["description"])
            self.assertTrue(callable(meta["function"]))


if __name__ == "__main__":
    unittest.main()
