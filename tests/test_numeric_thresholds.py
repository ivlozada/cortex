
import unittest
from cortex_omega.core.rules import Rule, Literal, RuleBase, FactBase, Scene
from cortex_omega.core.hypothesis import HypothesisGenerator, FailureContext
from cortex_omega.core.engine import infer

class TestNumericThresholds(unittest.TestCase):
    def test_numeric_split_discovery(self):
        """
        Scenario:
        - Target: is_senior(X)
        - Feature: age(X, V)
        - Pattern: age > 60 => is_senior
        - Current Rule: is_senior(X) :- (Empty) or (Wrong)
        - Error: FALSE_NEGATIVE for a senior.
        """
        
        # 1. Setup Memory (Positive Examples of Seniors)
        memory = []
        for i, age in enumerate([65, 70, 80, 62, 90]):
            fb = FactBase()
            fb.add("age", (f"p{i}", str(age)))
            scene = Scene(
                id=f"scene_pos_{i}",
                facts=fb,
                target_entity=f"p{i}",
                target_predicate="is_senior",
                ground_truth=True
            )
            memory.append(scene)
            
        # Add Negative Examples (Juniors)
        for i, age in enumerate([20, 30, 40, 50, 59]):
            fb = FactBase()
            fb.add("age", (f"n{i}", str(age)))
            scene = Scene(
                id=f"scene_neg_{i}",
                facts=fb,
                target_entity=f"n{i}",
                target_predicate="is_senior",
                ground_truth=False
            )
            memory.append(scene)
            
        # 2. Setup Failure Context
        # A new senior (age 75) is predicted False (FN)
        target_fb = FactBase()
        target_fb.add("age", ("target", "75"))
        
        # Initial Rule: Empty (or dummy)
        rule = Rule("R_senior", Literal("is_senior", ("X",)), [])
        
        ctx = FailureContext(
            rule=rule,
            error_type="FALSE_NEGATIVE",
            target_entity="target",
            target_predicate="is_senior",
            scene_facts=target_fb,
            memory=memory,
            prediction=False,
            ground_truth=True
        )
        
        # 3. Run Generator
        generator = HypothesisGenerator()
        candidates = generator.generate(ctx, top_k=5, beam_width=1)
        
        # 4. Verify Candidates
        found_threshold = False
        for patch, new_rule, aux in candidates:
            print(f"Candidate: {new_rule}")
            # Check for age(X, V) and V > T
            has_age = False
            has_gt = False
            threshold_val = 0.0
            
            for lit in new_rule.body:
                if lit.predicate == "age":
                    has_age = True
                if lit.predicate == ">":
                    has_gt = True
                    threshold_val = float(lit.args[1])
            
            if has_age and has_gt:
                # Threshold should be around 60 (between 59 and 62)
                if 59 <= threshold_val <= 62:
                    found_threshold = True
                    break
                    
        self.assertTrue(found_threshold, "Did not find a rule with age > ~60")

if __name__ == "__main__":
    unittest.main()
