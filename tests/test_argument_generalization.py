import unittest
from cortex_omega.core.rules import Term, Literal, Rule, RuleID, Scene, FactBase, parse_term
from cortex_omega.core.types import FailureContext, PatchOperation
from cortex_omega.core.strategies import ArgumentGeneralizationStrategy
from cortex_omega.core.config import KernelConfig

class TestArgumentGeneralization(unittest.TestCase):
    def test_generalization(self):
        config = KernelConfig()
        strategy = ArgumentGeneralizationStrategy(config)
        
        # Memory: add(0, 0, 0)
        scene1 = Scene(
            id="s1",
            facts=FactBase(),
            target_entity="dummy",
            target_predicate="add",
            ground_truth=True,
            target_args=(parse_term("zero"), parse_term("zero"), parse_term("zero"))
        )
        
        # Current Failure: add(0, s(0), s(0))
        # Should generalize to add(0, X, X)
        current_args = (parse_term("zero"), parse_term("s(zero)"), parse_term("s(zero)"))
        
        ctx = FailureContext(
            rule=Rule(RuleID("dummy"), Literal("add", ("X", "Y", "Z")), []),
            scene_facts=FactBase(),
            target_entity="dummy",
            target_predicate="add",
            ground_truth=True,
            prediction=False,
            error_type="FALSE_NEGATIVE",
            memory=[scene1],
            target_args=current_args
        )
        
        patches = strategy.propose(ctx, {})
        
        found = False
        for p in patches:
            rule = p.details["rule"]
            head = rule.head
            print(f"Proposed: {head}")
            # Expected: add(zero, X0, X0)
            # args[0] should be 'zero'
            # args[1] should be a var
            # args[2] should be SAME var
            
            if str(head.args[0]) == "zero" and \
               head.args[1] == head.args[2] and \
               head.args[1].startswith("X"):
                found = True
                
        self.assertTrue(found, "Should find add(zero, X, X)")

    def test_generalization_v2(self):
        config = KernelConfig()
        strategy = ArgumentGeneralizationStrategy(config)
        
        # Memory: add(0, 0, 0)
        scene1 = Scene(
            id="s1",
            facts=FactBase(),
            target_entity="dummy",
            target_predicate="add",
            ground_truth=True,
            target_args=(parse_term("zero"), parse_term("zero"), parse_term("zero"))
        )
        
        # Current Failure: add(s(0), 0, s(0))
        # Should generalize to add(X, 0, X)
        current_args = (parse_term("s(zero)"), parse_term("zero"), parse_term("s(zero)"))
        
        ctx = FailureContext(
            rule=Rule(RuleID("dummy"), Literal("add", ("X", "Y", "Z")), []),
            scene_facts=FactBase(),
            target_entity="dummy",
            target_predicate="add",
            ground_truth=True,
            prediction=False,
            error_type="FALSE_NEGATIVE",
            memory=[scene1],
            target_args=current_args
        )
        
        patches = strategy.propose(ctx, {})
        
        found = False
        for p in patches:
            rule = p.details["rule"]
            head = rule.head
            print(f"Proposed: {head}")
            
            if str(head.args[1]) == "zero" and \
               head.args[0] == head.args[2] and \
               head.args[0].startswith("X"):
                found = True
                
        self.assertTrue(found, "Should find add(X, zero, X)")

if __name__ == '__main__':
    unittest.main()
