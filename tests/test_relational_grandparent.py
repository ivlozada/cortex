
import unittest
from cortex_omega import Cortex
from cortex_omega.core.rules import Rule, Literal, FactBase, Scene

import logging
import sys
root = logging.getLogger()
root.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
root.addHandler(handler)
logger = logging.getLogger(__name__)

class TestRelationalGrandparent(unittest.TestCase):
    def test_grandparent_induction(self):
        """
        Scenario: Learn grandparent(X, Z) :- parent(X, Y), parent(Y, Z).
        """
        brain = Cortex()
        
        # Training Data
        # Family 1: A -> B -> C (A is grandparent of C)
        # Family 2: D -> E -> F (D is grandparent of F)
        # Family 3: G -> H (G is parent of H, not grandparent)
        
        data = [
            # Positive Examples (Grandparents)
            {
                "id": "scene_abc",
                "parent_ab": "parent(a, b)",
                "parent_bc": "parent(b, c)",
                "target_entity": "a",
                "target_args": ["a", "c"], # X=a, Z=c
                "is_grandparent": True
            },
            {
                "id": "scene_def",
                "parent_de": "parent(d, e)",
                "parent_ef": "parent(e, f)",
                "target_entity": "d",
                "target_args": ["d", "f"],
                "is_grandparent": True
            },
            # Negative Examples (Parents only)
            {
                "id": "scene_gh",
                "parent_gh": "parent(g, h)",
                "target_entity": "g",
                "target_args": ["g", "h"],
                "is_grandparent": False
            },
            # Negative Example (Disconnected / Wrong relation)
            {
                "id": "scene_ij",
                "friend_ij": "friend(i, j)",
                "target_entity": "i",
                "target_args": ["i", "j"],
                "is_grandparent": False
            }
        ]
        
        # We need to manually construct scenes because absorb_memory's default dict-to-fact 
        # mapping might not handle "parent(a, b)" string parsing automatically 
        # unless we use a specific format or helper.
        # Let's use the lower-level API or ensure absorb_memory handles it.
        # Actually, absorb_memory treats keys as predicates and values as objects.
        # So "parent": "b" for entity "a" means parent(a, b).
        # But here we have multiple parents.
        
        # Let's construct memory manually for precision in this test.
        memory = []
        
        # Scene 1: A->B->C
        fb1 = FactBase()
        fb1.add("parent", ("a", "b"))
        fb1.add("parent", ("b", "c"))
        s1 = Scene("s1", fb1, "a", "grandparent", ground_truth=True, target_args=("a", "c"))
        memory.append(s1)
        
        # Scene 2: D->E->F
        fb2 = FactBase()
        fb2.add("parent", ("d", "e"))
        fb2.add("parent", ("e", "f"))
        s2 = Scene("s2", fb2, "d", "grandparent", ground_truth=True, target_args=("d", "f"))
        memory.append(s2)
        
        # Scene 3: G->H (Negative)
        fb3 = FactBase()
        fb3.add("parent", ("g", "h"))
        s3 = Scene("s3", fb3, "g", "grandparent", ground_truth=False, target_args=("g", "h"))
        memory.append(s3)
        
        # Train
        from cortex_omega.core.engine import update_theory_kernel, KernelConfig
        
        theory = brain.theory
        memory_buffer = []
        config = KernelConfig()
        
        for s in memory:
            theory, memory_buffer = update_theory_kernel(theory, s, memory_buffer, brain.axioms, config)
            
        brain.theory = theory
        brain.memory = memory_buffer
            
        # Test: New Family X->Y->Z
        fb_test = FactBase()
        fb_test.add("parent", ("x", "y"))
        fb_test.add("parent", ("y", "z"))
        s_test = Scene("s_test", fb_test, "x", "grandparent", ground_truth=True, target_args=("x", "z"))
        
        from cortex_omega.core.engine import infer
        prediction, trace = infer(brain.theory, s_test)
        print(f"Prediction for X->Y->Z: {prediction}")
        
        # Inspect Rules
        rules = brain.inspect_rules("grandparent")
        print("Learned Rules:")
        found_recursive = False
        for r in rules:
            print(f"  - {r}")
            # Check for parent(X, Y), parent(Y, Z) structure
            # Variable names might vary, but structure is:
            # Body has 2 literals, both 'parent'.
            # Args chain: (A, B), (B, C) matching Head (A, C).
            
            if len(r.body) == 2 and r.body[0].predicate == "parent" and r.body[1].predicate == "parent":
                # Check variable chaining
                # Head: (V1, V2)
                # Body1: (V1, V3)
                # Body2: (V3, V2)
                
                head_vars = r.head.args
                b1_vars = r.body[0].args
                b2_vars = r.body[1].args
                
                if (b1_vars[0] == head_vars[0] and 
                    b2_vars[1] == head_vars[1] and 
                    b1_vars[1] == b2_vars[0]):
                    found_recursive = True
                    break
                    
        self.assertTrue(prediction, "Should predict grandparent for X->Y->Z")
        self.assertTrue(found_recursive, "Should learn relational rule: grandparent(X, Z) :- parent(X, Y), parent(Y, Z)")

if __name__ == "__main__":
    unittest.main()
