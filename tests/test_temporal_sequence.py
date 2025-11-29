
import unittest
from cortex_omega import Cortex
from cortex_omega.core.rules import Rule, Literal, FactBase, Scene
from cortex_omega.core.engine import update_theory_kernel, KernelConfig, infer

class TestTemporalSequence(unittest.TestCase):
    def test_temporal_sequence_learning(self):
        """
        Scenario: Learn risk(X) :- login(X, T1), reset_password(X, T2), T2 > T1.
        """
        brain = Cortex()
        
        # Training Data
        # Case 1: Risk (Login then Reset)
        fb1 = FactBase()
        fb1.add("login", ("user1", "10:00"))
        fb1.add("reset_password", ("user1", "10:05"))
        # We need a way to represent time comparison. 
        # Ideally, the engine should support built-in predicates like >(T1, T2).
        # For now, let's assume we provide facts or the engine can handle string/int comparison if we enable it.
        # Let's use integer timestamps for simplicity.
        fb1 = FactBase()
        fb1.add("login", ("user1", "100"))
        fb1.add("reset_password", ("user1", "105"))
        s1 = Scene("s1", fb1, "user1", "risk", ground_truth=True)
        
        # Case 2: Risk (Login then Reset)
        fb2 = FactBase()
        fb2.add("login", ("user2", "200"))
        fb2.add("reset_password", ("user2", "210"))
        s2 = Scene("s2", fb2, "user2", "risk", ground_truth=True)
        
        # Case 3: No Risk (Reset then Login - weird but order matters)
        fb3 = FactBase()
        fb3.add("reset_password", ("user3", "300"))
        fb3.add("login", ("user3", "305"))
        s3 = Scene("s3", fb3, "user3", "risk", ground_truth=False)
        
        # Case 4: No Risk (Login only)
        fb4 = FactBase()
        fb4.add("login", ("user4", "400"))
        s4 = Scene("s4", fb4, "user4", "risk", ground_truth=False)
        
        # Additional Positive Cases to force generalization (MDL)
        fb5 = FactBase()
        fb5.add("login", ("user5", "500"))
        fb5.add("reset_password", ("user5", "510"))
        s5 = Scene("s5", fb5, "user5", "risk", ground_truth=True)
        
        fb6 = FactBase()
        fb6.add("login", ("user6", "600"))
        fb6.add("reset_password", ("user6", "610"))
        s6 = Scene("s6", fb6, "user6", "risk", ground_truth=True)
        
        # Additional Negative Case to break specific negative rules
        fb7 = FactBase()
        fb7.add("reset_password", ("user7", "700"))
        fb7.add("login", ("user7", "705"))
        s7 = Scene("s7", fb7, "user7", "risk", ground_truth=False)
        
        memory = [s1, s2, s3, s4, s5, s6, s7]
        
        # Train
        theory = brain.theory
        memory_buffer = []
        config = KernelConfig()
        config.lambda_complexity = 0.5 # Force generalization over memorization
        
        # We need to ensure the engine has a strategy to propose "T2 > T1".
        # This might require a new strategy: _strategy_temporal_ordering
        
        for s in memory:
            theory, memory_buffer = update_theory_kernel(theory, s, memory_buffer, brain.axioms, config)
            
        brain.theory = theory
        
        # Test: New Risk Case
        fb_test = FactBase()
        fb_test.add("login", ("user_test", "500"))
        fb_test.add("reset_password", ("user_test", "505"))
        s_test = Scene("s_test", fb_test, "user_test", "risk", ground_truth=True)
        
        prediction, trace = infer(brain.theory, s_test)
        print(f"Prediction for Test Case: {prediction}")
        
        # Inspect Rules
        rules = brain.inspect_rules("risk")
        print("Learned Rules:")
        found_temporal = False
        for r in rules:
            print(f"  - {r}")
            # Check for structure: login(X, T1), reset(X, T2), >(T2, T1)
            # Or similar.
            preds = [lit.predicate for lit in r.body]
            if "login" in preds and "reset_password" in preds:
                # Check for comparison
                for lit in r.body:
                    if lit.predicate == ">":
                        found_temporal = True
                        
        self.assertTrue(prediction, "Should predict risk for sequential events")
        self.assertTrue(found_temporal, "Should learn temporal constraint T2 > T1")

if __name__ == "__main__":
    unittest.main()
