import unittest
from cortex_omega.core.rules import Rule, Literal, FactBase, Scene, RuleID, make_rule_id

class TestRepresentation(unittest.TestCase):
    def test_literal_equality(self):
        l1 = Literal("p", ("a", "b"))
        l2 = Literal("p", ("a", "b"))
        l3 = Literal("p", ("a", "c"))
        l4 = Literal("q", ("a", "b"))
        l5 = Literal("p", ("a", "b"), negated=True)
        
        self.assertEqual(l1, l2)
        self.assertNotEqual(l1, l3)
        self.assertNotEqual(l1, l4)
        self.assertNotEqual(l1, l5)
        
        self.assertEqual(hash(l1), hash(l2))
        
    def test_rule_id_determinism(self):
        head = Literal("target", ("X",))
        body = [Literal("p", ("X", "Y")), Literal("q", ("Y", "Z"))]
        
        r1 = Rule(RuleID("temp"), head, body)
        id1 = make_rule_id(r1)
        
        r2 = Rule(RuleID("temp2"), head, body)
        id2 = make_rule_id(r2)
        
        self.assertEqual(id1, id2)
        self.assertEqual(str(id1), str(id2))
        
        # Change body order (should be same ID due to sorting)
        body_reversed = [Literal("q", ("Y", "Z")), Literal("p", ("X", "Y"))]
        r3 = Rule(RuleID("temp3"), head, body_reversed)
        id3 = make_rule_id(r3)
        
        self.assertEqual(id1, id3)
        
    def test_factbase_query(self):
        fb = FactBase()
        fb.add("p", ("a", "b"))
        fb.add("p", ("b", "c"))
        fb.add("q", ("a",))
        
        self.assertEqual(len(fb.query("p")), 2)
        self.assertEqual(len(fb.query("q")), 1)
        self.assertEqual(len(fb.query("z")), 0)
        
        # Query with wildcard
        res = fb.query("p", ("a", None))
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0], ("a", "b"))
        
    def test_scene_immutability(self):
        # Scene should be treated as immutable for ID purposes usually, 
        # but Python doesn't enforce it.
        # Just check basic properties.
        fb = FactBase()
        s = Scene("s1", fb, "a", "target", True)
        self.assertEqual(s.target_args, ("a",)) # Default args
        
if __name__ == "__main__":
    unittest.main()
