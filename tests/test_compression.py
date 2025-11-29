
import unittest
from cortex_omega.core.rules import Rule, Literal, RuleBase
from cortex_omega.core.engine import prune_redundant_rules

class TestCompression(unittest.TestCase):
    def test_subsumption_logic(self):
        # General: p(X) :- q(X)
        r_gen = Rule("Gen", Literal("p", ("X",)), [Literal("q", ("X",))])
        
        # Specific: p(X) :- q(X), r(X)
        r_spec = Rule("Spec", Literal("p", ("X",)), [Literal("q", ("X",)), Literal("r", ("X",))])
        
        # Different Head: z(X) :- q(X)
        r_diff = Rule("Diff", Literal("z", ("X",)), [Literal("q", ("X",))])
        
        self.assertTrue(r_spec.is_subsumed_by(r_gen))
        self.assertFalse(r_gen.is_subsumed_by(r_spec))
        self.assertFalse(r_spec.is_subsumed_by(r_diff))

    def test_prune_redundant(self):
        theory = RuleBase()
        
        # Scenario 1: General rule is reliable -> Prune specific
        r_gen = Rule("Gen", Literal("p", ("X",)), [Literal("q", ("X",))])
        r_gen.fires_pos = 100
        r_gen.fires_neg = 0 # 100% reliable
        
        r_spec = Rule("Spec", Literal("p", ("X",)), [Literal("q", ("X",)), Literal("r", ("X",))])
        r_spec.fires_pos = 10
        r_spec.fires_neg = 0 # 100% reliable
        
        theory.add(r_gen)
        theory.add(r_spec)
        
        prune_redundant_rules(theory)
        
        self.assertIn("Gen", theory.rules)
        self.assertNotIn("Spec", theory.rules)
        
    def test_keep_exception(self):
        theory = RuleBase()
        
        # Scenario 2: General rule is unreliable -> Keep specific (Exception)
        r_gen = Rule("Gen", Literal("p", ("X",)), [Literal("q", ("X",))])
        r_gen.fires_pos = 50
        r_gen.fires_neg = 50 # 50% reliable (Bad)
        
        r_spec = Rule("Spec", Literal("p", ("X",)), [Literal("q", ("X",)), Literal("r", ("X",))])
        r_spec.fires_pos = 10
        r_spec.fires_neg = 0 # 100% reliable (Good Exception)
        
        theory.add(r_gen)
        theory.add(r_spec)
        
        prune_redundant_rules(theory)
        
        self.assertIn("Gen", theory.rules) # We don't prune general rules based on specific ones here
        self.assertIn("Spec", theory.rules) # Specific should survive because General is weak

if __name__ == "__main__":
    unittest.main()
