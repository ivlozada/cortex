import unittest
from cortex_omega.core.rules import Rule, Literal, FactBase, Scene, RuleID
from cortex_omega.core.hypothesis import HeuristicGenerator, FailureContext, PatchOperation

class TestAntiUnification(unittest.TestCase):
    def test_relational_anti_unification(self):
        # 1. Setup Scenes
        # Scene 1: A -> B -> C (Current Failure)
        fb1 = FactBase()
        fb1.add("parent", ("a", "b"))
        fb1.add("parent", ("b", "c"))
        s1 = Scene("s1", fb1, "a", "grandparent", ground_truth=True, target_args=("a", "c"))
        
        # Scene 2: D -> E -> F (Memory Example)
        fb2 = FactBase()
        fb2.add("parent", ("d", "e"))
        fb2.add("parent", ("e", "f"))
        s2 = Scene("s2", fb2, "d", "grandparent", ground_truth=True, target_args=("d", "f"))
        
        # 2. Create Context
        ctx = FailureContext(
            rule=None, # No rule yet
            error_type="FALSE_NEGATIVE",
            target_entity="a",
            scene_facts=fb1,
            prediction=False,
            ground_truth=True,
            target_predicate="grandparent",
            target_args=("a", "c"),
            memory=[s2]
        )
        
        # 3. Run Strategy
        gen = HeuristicGenerator()
        patches = gen._strategy_relational_anti_unification(ctx, features={})
        
        # 4. Verify
        self.assertTrue(len(patches) > 0, "Should generate at least one patch")
        
        found_transitive = False
        for p in patches:
            if p.operation == PatchOperation.CREATE_BRANCH:
                rule = p.details["rule"]
                # Check body: parent(X, Y), parent(Y, Z)
                if len(rule.body) == 2:
                    l1 = rule.body[0]
                    l2 = rule.body[1]
                    if l1.predicate == "parent" and l2.predicate == "parent":
                        # Check variable chaining
                        # Head: (X, Z)
                        # Body: (X, Y), (Y, Z)
                        if l1.args[1] == l2.args[0]:
                            found_transitive = True
                            break
                            
        self.assertTrue(found_transitive, "Should find transitive rule: parent(X, Y), parent(Y, Z)")

if __name__ == "__main__":
    unittest.main()
