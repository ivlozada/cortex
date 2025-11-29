
import unittest
import time
from cortex_omega.api.client import Cortex
from cortex_omega.core.rules import Rule, Literal

class TestEnginePolish(unittest.TestCase):
    def test_rule_timestamps(self):
        """Verify Rule objects have timestamps."""
        r = Rule("test_rule", Literal("head", ("X",)), [])
        self.assertGreater(r.created_at, 0)
        self.assertGreater(r.updated_at, 0)
        
        # Test serialization
        d = r.to_dict()
        self.assertIn("meta", d)
        self.assertIn("created_at", d["meta"])
        
    def test_config_knobs(self):
        """Verify Cortex accepts new config knobs."""
        brain = Cortex(
            sensitivity=0.5,
            mode="strict",
            priors={"rule_base": 0.8, "exception": 0.1},
            noise_model={"false_positive": 0.1, "false_negative": 0.1},
            plasticity={"min_conf_to_keep": 0.9, "max_rule_count": 100},
            feature_priors={"important_feature": 10.0}
        )
        
        self.assertEqual(brain.config.mode, "strict")
        self.assertEqual(brain.config.priors["rule_base"], 0.8)
        self.assertEqual(brain.config.feature_priors["important_feature"], 10.0)
        self.assertEqual(brain.config.plasticity["min_conf_to_keep"], 0.9)
        
    def test_inspect_rules(self):
        """Verify inspect_rules returns rich objects."""
        brain = Cortex()
        brain.absorb_memory([
            {"feature": "A", "target": "true"},
            {"feature": "A", "target": "true"},
        ], target_label="target")
        
        rules = brain.inspect_rules()
        self.assertGreater(len(rules), 0)
        r = rules[0]
        self.assertIsInstance(r, Rule)
        self.assertTrue(hasattr(r, "created_at"))
        
    def test_feature_priors_integration(self):
        """
        Verify feature priors are passed to the engine.
        We can't easily check internal scoring without mocking, 
        but we can check if the config is correctly set in the kernel.
        """
        brain = Cortex(feature_priors={"color": 2.0})
        self.assertEqual(brain.config.feature_priors["color"], 2.0)
        
        # Run a simple learning task to ensure no crash
        brain.absorb_memory([
            {"color": "red", "shape": "square", "target": "true"},
            {"color": "blue", "shape": "circle", "target": "false"},
        ], target_label="target")
        
        # If it didn't crash, the plumbing is likely correct.

if __name__ == '__main__':
    unittest.main()
