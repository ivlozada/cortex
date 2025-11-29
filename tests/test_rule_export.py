
import unittest
import json
from cortex_omega.api.client import Cortex

class TestRuleExport(unittest.TestCase):
    def test_export_json(self):
        """
        Verify JSON export format.
        """
        brain = Cortex()
        # Train a simple rule
        brain.absorb_memory([
            {"feature": "A", "target": "true"},
            {"feature": "A", "target": "true"},
        ], target_label="target")
        
        json_str = brain.export_rules(format="json")
        data = json.loads(json_str)
        
        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 0)
        rule = data[0]
        self.assertIn("id", rule)
        self.assertIn("head", rule)
        self.assertIn("body", rule)
        self.assertIn("stats", rule)
        self.assertIn("reliability", rule["stats"])
        
    def test_export_prolog(self):
        """
        Verify Prolog export format.
        """
        brain = Cortex()
        brain.absorb_memory([
            {"feature": "A", "target": "true"},
            {"feature": "A", "target": "true"},
        ], target_label="target")
        
        prolog_str = brain.export_rules(format="prolog")
        
        self.assertIsInstance(prolog_str, str)
        self.assertIn(":-", prolog_str)
        self.assertTrue(prolog_str.endswith("."))
        
        # Check specific syntax
        # target(X) :- feature(X, A).
        self.assertIn("target(", prolog_str)
        self.assertIn("feature(", prolog_str)

if __name__ == '__main__':
    unittest.main()
