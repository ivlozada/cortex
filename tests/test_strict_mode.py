
import unittest
from cortex_omega.api.client import Cortex
from cortex_omega.core.engine import KernelConfig
from cortex_omega.core.rules import Rule, Literal

class TestStrictMode(unittest.TestCase):
    def test_strict_mode_punishment(self):
        """
        Verify that in STRICT mode, a single counter-example kills a rule (confidence -> 0.0).
        """
        brain = Cortex()
        
        # 1. Train a rule: "Balsa floats"
        train_data = [
            {"material": "balsa", "floats": "true"},
            {"material": "balsa", "floats": "true"},
            {"material": "balsa", "floats": "true"},
        ]
        brain.absorb_memory(train_data, target_label="floats")
        
        # Verify rule exists and has high confidence
        rules = brain.inspect_rules("floats")
        balsa_rule = None
        for r in rules:
            if "material(X, balsa)" in str(r):
                balsa_rule = r
                break
        
        self.assertIsNotNone(balsa_rule)
        self.assertGreater(balsa_rule.confidence, 0.8)
        
        # 2. Switch to STRICT mode
        # We need to access the kernel config directly or via API if exposed.
        # For now, we'll modify the internal config if possible, or pass it if API supports.
        # Assuming we can set it on the brain or pass it to absorb/infer.
        # Since API doesn't expose config directly yet, we might need to hack it or add API support.
        # Let's assume we add `brain.set_mode("STRICT")`.
        
        if hasattr(brain, "set_mode"):
            brain.set_mode("STRICT")
        else:
            # Fallback: Modify internal config if accessible (it's not easily accessible in current client)
            # So we will implement set_mode in client.py as part of this task.
            pass

        # 3. Provide a counter-example: "Balsa that sinks"
        counter_example = [{"material": "balsa", "floats": "false"}]
        brain.absorb_memory(counter_example, target_label="floats")
        
        # Verify rule is killed
        killed_rule = brain.theory.rules.get(balsa_rule.id)
        print(f"DEBUG: Killed Rule Confidence: {killed_rule.confidence}")
        
        # In STRICT mode, confidence should be ~0.0 (due to massive failure count)
        self.assertLess(killed_rule.confidence, 0.01)

if __name__ == '__main__':
    unittest.main()
