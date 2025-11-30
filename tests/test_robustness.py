import unittest
from cortex_omega import Cortex
import sys
print(f"DEBUG: sys.path: {sys.path}")
from cortex_omega.core.rules import Literal, Rule
import logging

# logging.basicConfig(level=logging.DEBUG)
# logging.getLogger("cortex_omega").setLevel(logging.DEBUG)

class TestRobustness(unittest.TestCase):
    def setUp(self):
        # CORTEX-OMEGA v1.3: Use "strict" mode for this test
        # This test expects a single counter-example to override a general rule.
        self.brain = Cortex(mode="strict")

    def test_conflict_resolution_david_vs_goliath(self):
        """
        Test that specific exceptions override general rules.
        """
        # General Rule: Heavy objects sink
        self.brain.add_rule("sink(X) :- heavy(X)")
        
        # Specific Rule: Balsa wood floats (even if heavy-ish context)
        # Note: In our logic, we need to ensure the specific rule fires and we can distinguish.
        # Here we simulate it by training.
        
        # Train General
        self.brain.absorb_memory([
            {"id": "iron", "heavy": True, "sink": True},
            {"id": "stone", "heavy": True, "sink": True}
        ], target_label="sink")
        
        # Verify General
        res = self.brain.query(id="lead", heavy=True, target="sink")
        self.assertTrue(res.prediction, "General rule should predict sink")
        
        # Train Exception (Balsa)
        # We need a feature that distinguishes balsa, e.g., 'wood'
        self.brain.absorb_memory([
            {"id": "balsa", "heavy": True, "wood": True, "sink": False}
        ], target_label="sink")
        
        # Verify Exception
        res_balsa = self.brain.query(id="balsa_test", heavy=True, wood=True, target="sink")
        self.assertFalse(res_balsa.prediction, "Exception should NOT sink")
        
        # Verify General still holds
        # Verify General still holds
        res_iron = self.brain.query(id="iron_test", heavy=True, wood=False, target="sink")
        self.assertTrue(res_iron.prediction, "General rule should still sink non-wood")

    def test_plasticity_kill_switch(self):
        """
        Test that we can unlearn a rule instantly.
        """
        # Teach a rule
        self.brain.add_rule("fly(X) :- bird(X)")
        
        # Verify it works
        res = self.brain.query(id="pigeon", bird=True, target="fly")
        self.assertTrue(res.prediction)
        
        # Context Shift (Kill Switch)
        self.brain.context_shift("No Fly Zone")
        
        # Verify confidence degraded
        # We need to access the rule to check confidence, or check query confidence
        res_after = self.brain.query(id="pigeon", bird=True, target="fly")
        # It might still predict True but with lower confidence, or False if threshold met.
        # Our context_shift halves confidence.
        
        # Let's check internal rule confidence
        rule = list(self.brain.theory.rules.values())[0]
        self.assertLess(rule.confidence, 1.0, "Confidence should have decayed")

    def test_noise_robustness(self):
        """
        Test that the engine ignores irrelevant features.
        """
        # Data where 'color' is noise, 'shape' is signal
        data = [
            {"id": "1", "shape": "square", "color": "red", "is_target": True},
            {"id": "2", "shape": "square", "color": "blue", "is_target": True},
            {"id": "3", "shape": "circle", "color": "red", "is_target": False},
            {"id": "4", "shape": "circle", "color": "blue", "is_target": False},
        ]
        
        self.brain.absorb_memory(data, target_label="target")
        
        # Query with new noise value
        res = self.brain.query(id="test", shape="square", color="green", target="target")
        self.assertTrue(res.prediction, "Should predict True based on shape, ignoring color")
        
        # Check explanation to ensure it uses 'shape'
        self.assertIn("shape", str(res.explanation), "Explanation should cite shape")

if __name__ == '__main__':
    unittest.main()
