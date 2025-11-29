
import unittest
import random
import copy
from cortex_omega.api.client import Cortex

class TestProperties(unittest.TestCase):
    def test_permutation_invariance(self):
        """
        Property: The order of training examples should not affect the final accuracy 
        (though it might affect the exact rule ID or path).
        """
        data = []
        # Generate synthetic data: target = (A=1 AND B=1)
        for _ in range(50):
            a = random.choice(["0", "1"])
            b = random.choice(["0", "1"])
            target = "true" if a == "1" and b == "1" else "false"
            data.append({"A": a, "B": b, "target": target})
            
        # Run 1
        brain1 = Cortex(sensitivity=0.1)
        brain1.absorb_memory(data, target_label="target")
        
        # Run 2 (Shuffled)
        data2 = copy.deepcopy(data)
        random.shuffle(data2)
        brain2 = Cortex(sensitivity=0.1)
        brain2.absorb_memory(data2, target_label="target")
        
        # Verify both learned the concept
        test_case = {"A": "1", "B": "1"}
        res1 = brain1.query(A="1", B="1", target="target")
        res2 = brain2.query(A="1", B="1", target="target")
        
        self.assertTrue(res1.prediction)
        self.assertTrue(res2.prediction)
        
    def test_noise_invariance(self):
        """
        Property: Adding a purely random feature should not change the prediction.
        """
        data = []
        for _ in range(50):
            a = random.choice(["0", "1"])
            target = "true" if a == "1" else "false"
            data.append({"A": a, "target": target})
            
        brain = Cortex(sensitivity=0.1)
        brain.absorb_memory(data, target_label="target")
        
        # Baseline prediction
        res_base = brain.query(A="1", target="target")
        self.assertTrue(res_base.prediction)
        
        # Add noise feature to query
        res_noise = brain.query(A="1", NOISE="random_val", target="target")
        self.assertEqual(res_base.prediction, res_noise.prediction)
        self.assertAlmostEqual(res_base.confidence, res_noise.confidence, delta=0.1)

if __name__ == '__main__':
    unittest.main()
