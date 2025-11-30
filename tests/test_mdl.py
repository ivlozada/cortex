
import unittest
from cortex_omega.core.rules import Rule, Literal, RuleBase
from cortex_omega.core.critic import Critic, garbage_collect
from cortex_omega.core.config import KernelConfig

class TestMDL(unittest.TestCase):
    def test_score_mdl(self):
        # Rule A: Simple (Complexity 2)
        # p(X) :- q(X)
        rule_a = Rule("A", Literal("p", ("X",)), [Literal("q", ("X",))])
        rule_a.fires_pos = 10
        rule_a.fires_neg = 0
        # Score = 10 - 0 - 0.2 * 2 = 9.6
        
        # Rule B: Complex (Complexity 4)
        # p(X) :- q(X), r(X), s(X)
        rule_b = Rule("B", Literal("p", ("X",)), [Literal("q", ("X",)), Literal("r", ("X",)), Literal("s", ("X",))])
        rule_b.fires_pos = 10
        rule_b.fires_neg = 0
        # Score = 10 - 0 - 0.2 * 4 = 9.2
        
        critic = Critic(KernelConfig())
        score_a = critic.score_mdl(rule_a)
        score_b = critic.score_mdl(rule_b)
        
        print(f"Score A: {score_a}, Score B: {score_b}")
        self.assertGreater(score_a, score_b)
        
    def test_garbage_collect_mdl(self):
        theory = RuleBase()
        
        # Rule Good: High Pos, Low Neg
        rule_good = Rule("Good", Literal("p", ("X",)), [Literal("q", ("X",))])
        rule_good.fires_pos = 20
        rule_good.fires_neg = 1
        rule_good.support_count = 21
        theory.add(rule_good)
        
        # Rule Bad: Low Pos, High Neg
        rule_bad = Rule("Bad", Literal("p", ("X",)), [Literal("z", ("X",))])
        rule_bad.fires_pos = 5
        rule_bad.fires_neg = 10 # -5 net
        rule_bad.support_count = 15
        theory.add(rule_bad)
        
        # Rule New: Low stats but low support (Grace Period)
        rule_new = Rule("New", Literal("p", ("X",)), [Literal("n", ("X",))])
        rule_new.fires_pos = 0
        rule_new.fires_neg = 1
        rule_new.support_count = 1 # < 5
        theory.add(rule_new)
        
        garbage_collect(theory, threshold=0.0)
        
        self.assertIn("Good", theory.rules)
        self.assertNotIn("Bad", theory.rules)
        self.assertIn("New", theory.rules) # Grace period

if __name__ == "__main__":
    unittest.main()
