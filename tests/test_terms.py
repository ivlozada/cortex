import unittest
from cortex_omega.core.rules import Term, Literal, parse_term
from cortex_omega.core.inference import InferenceEngine
from cortex_omega.core.rules import FactBase, RuleBase

class TestTerms(unittest.TestCase):
    def test_term_creation(self):
        t1 = Term("s", (Term("0"),))
        self.assertEqual(str(t1), "s(0)")
        
        t2 = parse_term("s(s(0))")
        self.assertEqual(t2.name, "s")
        self.assertEqual(len(t2.args), 1)
        self.assertEqual(t2.args[0].name, "s")
        self.assertEqual(str(t2), "s(s(0))")

    def test_recursive_unification(self):
        engine = InferenceEngine(FactBase(), RuleBase())
        
        # Unify s(X) with s(0)
        # Literal: p(s(X))
        # Fact: p(s(0))
        
        lit = Literal("p", (Term("s", (Term("X"),)),))
        fact_args = (Term("s", (Term("0"),)),)
        
        bindings = engine.unify(lit, fact_args)
        self.assertIsNotNone(bindings)
        self.assertEqual(bindings["X"], Term("0"))
        
    def test_deep_unification(self):
        engine = InferenceEngine(FactBase(), RuleBase())
        
        # Unify add(s(X), Y, s(Z)) with add(s(0), s(0), s(s(0)))
        # Should bind X->0, Y->s(0), Z->s(0)
        
        lit = Literal("add", (
            parse_term("s(X)"),
            parse_term("Y"),
            parse_term("s(Z)")
        ))
        
        fact_args = (
            parse_term("s(0)"),
            parse_term("s(0)"),
            parse_term("s(s(0))")
        )
        
        bindings = engine.unify(lit, fact_args)
        self.assertIsNotNone(bindings)
        self.assertEqual(bindings["X"], Term("0"))
        self.assertEqual(bindings["Y"], parse_term("s(0)"))
        self.assertEqual(bindings["Z"], parse_term("s(0)"))

    def test_fail_unification(self):
        engine = InferenceEngine(FactBase(), RuleBase())
        
        # Unify s(X) with 0 -> Fail
        lit = Literal("p", (parse_term("s(X)"),))
        fact_args = (parse_term("0"),)
        
        bindings = engine.unify(lit, fact_args)
        self.assertIsNone(bindings)

if __name__ == '__main__':
    unittest.main()
