import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "mini_diffusion" / "evaluate_comparison.py"


class EvaluationLedgerIntegrationTests(unittest.TestCase):
    def test_evaluator_routes_append_through_helper_api(self):
        source = SOURCE.read_text(encoding="utf-8")
        tree = ast.parse(source)
        imports = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            and node.module == "tools.experiment_ledger"
            for alias in node.names
        }
        self.assertIn("append_event", imports)
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "append_experiment_event"
        ]
        self.assertEqual(len(calls), 1)
        self.assertNotIn('ledger.open("a"', source)
        self.assertNotIn("reports/experiment_ledger.jsonl", source)
        self.assertNotIn('"event_id": event_id', source)
        self.assertNotIn('"timestamp_utc": datetime.now', source)


if __name__ == "__main__":
    unittest.main()
