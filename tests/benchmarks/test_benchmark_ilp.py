
import unittest
import time
from cortex_omega.api.client import Cortex

class BenchmarkILP(unittest.TestCase):
    def test_benchmark_xor(self):
        """Learn XOR function."""
        data = [
            {"A": "0", "B": "0", "target": "false"},
            {"A": "0", "B": "1", "target": "true"},
            {"A": "1", "B": "0", "target": "true"},
            {"A": "1", "B": "1", "target": "false"},
        ]
        # Repeat to give enough signal
        data = data * 5
        
        start = time.time()
        brain = Cortex(sensitivity=0.01)
        brain.absorb_memory(data, target_label="target")
        end = time.time()
        
        print(f"\n[Benchmark] XOR Time: {end - start:.4f}s")
        
        # Verify accuracy
        correct = 0
        for case in data[:4]:
            res = brain.query(A=case["A"], B=case["B"], target="target")
            pred = "true" if res.prediction else "false"
            if pred == case["target"]:
                correct += 1
        
        print(f"[Benchmark] XOR Accuracy: {correct/4:.2f}")
        self.assertEqual(correct, 4)

    def test_benchmark_grandparent(self):
        """Learn Grandparent relation (Relational)."""
        from cortex_omega.core.rules import FactBase, Scene
        from cortex_omega.core.engine import KernelConfig
        from cortex_omega.core.learner import Learner
        
        # Construct relational memory manually
        memory = []
        
        # Family 1: A->B->C
        fb1 = FactBase()
        fb1.add("parent", ("a", "b"))
        fb1.add("parent", ("b", "c"))
        s1 = Scene("s1", fb1, "a", "grandparent", ground_truth=True, target_args=("a", "c"))
        memory.append(s1)
        
        # Family 2: D->E->F
        fb2 = FactBase()
        fb2.add("parent", ("d", "e"))
        fb2.add("parent", ("e", "f"))
        s2 = Scene("s2", fb2, "d", "grandparent", ground_truth=True, target_args=("d", "f"))
        memory.append(s2)
        
        # Negative: G->H
        fb3 = FactBase()
        fb3.add("parent", ("g", "h"))
        s3 = Scene("s3", fb3, "g", "grandparent", ground_truth=False, target_args=("g", "h"))
        memory.append(s3)
        
        start = time.time()
        brain = Cortex(sensitivity=0.1)
        
        # Manually train kernel
        theory = brain.theory
        memory_buffer = []
        config = brain.config
        
        learner = Learner(config)
        for s in memory:
            theory, memory_buffer = learner.learn(theory, s, memory_buffer, brain.axioms)
            
        brain.theory = theory
        end = time.time()
        
        print(f"\n[Benchmark] Grandparent Time: {end - start:.4f}s")
        
        # Test generalization: Chronos->Zeus->Hercules
        fb_test = FactBase()
        fb_test.add("parent", ("chronos", "zeus"))
        fb_test.add("parent", ("zeus", "hercules"))
        s_test = Scene("s_test", fb_test, "chronos", "grandparent", ground_truth=True, target_args=("chronos", "hercules"))
        
        from cortex_omega.core.engine import infer
        prediction, trace = infer(brain.theory, s_test)
        
        print(f"[Benchmark] Grandparent Generalization: {prediction}")
        self.assertTrue(prediction)

if __name__ == '__main__':
    unittest.main()
