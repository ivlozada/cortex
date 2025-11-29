
import unittest
from cortex_omega.core.rules import Rule, Literal, FactBase, Scene
from cortex_omega.core.hypothesis import HypothesisGenerator, FailureContext, DiscriminativeFeatureSelector

class TestConfounderInvariance(unittest.TestCase):
    def test_stability_check(self):
        """
        Scenario:
        - Feature 'color=red' is a confounder:
            - Early: Highly correlated with Target
            - Late: Anti-correlated or Uncorrelated
        - Feature 'shape=square' is causal:
            - Early: Correlated
            - Late: Correlated
        
        Expectation: Selector should score 'shape' higher than 'color'.
        """
        memory = []
        
        # Phase 1: Early (Red is correlated, Square is correlated)
        # 10 examples
        for i in range(10):
            fb = FactBase()
            fb.add("color", (f"e{i}", "red"))
            fb.add("shape", (f"e{i}", "square"))
            scene = Scene(
                id=f"early_{i}",
                facts=fb,
                target_entity=f"e{i}",
                target_predicate="glows",
                ground_truth=True
            )
            memory.append(scene)
            
        # Phase 2: Late (Red is uncorrelated/negative, Square is still correlated)
        # 10 examples where Red -> False, Square -> True
        for i in range(10):
            # Case A: Red but NOT Square -> False
            fb_a = FactBase()
            fb_a.add("color", (f"l_a{i}", "red"))
            fb_a.add("shape", (f"l_a{i}", "circle"))
            scene_a = Scene(
                id=f"late_a_{i}",
                facts=fb_a,
                target_entity=f"l_a{i}",
                target_predicate="glows",
                ground_truth=False
            )
            memory.append(scene_a)
            
            # Case B: Blue but Square -> True
            fb_b = FactBase()
            fb_b.add("color", (f"l_b{i}", "blue"))
            fb_b.add("shape", (f"l_b{i}", "square"))
            scene_b = Scene(
                id=f"late_b_{i}",
                facts=fb_b,
                target_entity=f"l_b{i}",
                target_predicate="glows",
                ground_truth=True
            )
            memory.append(scene_b)
            
        # Context: We need to select features for "glows"
        # We create a dummy context
        ctx = FailureContext(
            rule=Rule("dummy", Literal("glows", ("X",)), []),
            error_type="FALSE_NEGATIVE",
            target_entity="dummy",
            target_predicate="glows",
            scene_facts=FactBase(),
            memory=memory,
            prediction=False,
            ground_truth=True
        )
        
        selector = DiscriminativeFeatureSelector(min_score=0.0)
        # We need to access the internal scoring logic or just check the output order.
        # select_features returns a list of predicates.
        
        selected = selector.select_features(ctx)
        print(f"Selected Features: {selected}")
        
        # 'shape' should be in the list and 'color' should be lower or absent
        self.assertIn("shape", selected)
        
        # To verify scores, we can peek at the implementation or trust the order.
        # But select_features sorts by score.
        if "color" in selected:
            idx_shape = selected.index("shape")
            idx_color = selected.index("color")
            self.assertLess(idx_shape, idx_color, "Shape should be ranked higher than Color")
        else:
            # Color was filtered out (score too low)
            pass

if __name__ == "__main__":
    unittest.main()
