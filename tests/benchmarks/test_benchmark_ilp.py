
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
        brain = Cortex(sensitivity=0.1)
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
        """Learn Grandparent relation."""
        data = [
            # Family 1
            {"father": "abraham", "child": "isaac", "target": "false"},
            {"father": "isaac", "child": "jacob", "target": "false"},
            {"grandparent": "abraham", "grandchild": "jacob", "target": "true"},
            
            # Family 2
            {"father": "tywin", "child": "cersei", "target": "false"},
            {"father": "cersei", "child": "joffrey", "target": "false"},
            {"grandparent": "tywin", "grandchild": "joffrey", "target": "true"},
        ]
        
        start = time.time()
        brain = Cortex(sensitivity=0.1)
        brain.absorb_memory(data, target_label="target")
        end = time.time()
        
        print(f"\n[Benchmark] Grandparent Time: {end - start:.4f}s")
        
        # Test generalization
        # If we add a new chain, does it infer?
        brain.absorb_memory([
            {"father": "chronos", "child": "zeus", "target": "false"},
            {"father": "zeus", "child": "hercules", "target": "false"},
        ], target_label="target")
        
        res = brain.query(grandparent="chronos", grandchild="hercules", target="target")
        print(f"[Benchmark] Grandparent Generalization: {res.prediction}")
        self.assertTrue(res.prediction)

if __name__ == '__main__':
    unittest.main()
