import unittest
import pickle
import tempfile
import os
from cortex_omega import Cortex

class TestSerialization(unittest.TestCase):
    def setUp(self):
        self.brain = Cortex()
        self.brain.add_rule("mortal(X) :- human(X)")
        self.brain.absorb_memory([{"id": "socrates", "human": True, "mortal": True}], target_label="mortal")

    def test_pickle_round_trip(self):
        """
        Test that the brain can be pickled and unpickled without losing state.
        """
        # Verify initial state
        print(f"DEBUG: Facts in brain: {self.brain.facts.facts}")
        res_before = self.brain.query(id="socrates", target="mortal")
        print(f"DEBUG: Prediction: {res_before.prediction}")
        self.assertTrue(res_before.prediction)
        
        # Pickle
        with tempfile.NamedTemporaryFile(delete=False) as f:
            pickle.dump(self.brain, f)
            temp_path = f.name
            
        try:
            # Unpickle
            with open(temp_path, "rb") as f:
                brain_loaded = pickle.load(f)
                
            # Verify loaded state
            res_after = brain_loaded.query(id="socrates", target="mortal")
            self.assertTrue(res_after.prediction)
            self.assertEqual(res_after.explanation, res_before.explanation)
            
        finally:
            os.remove(temp_path)

if __name__ == '__main__':
    unittest.main()
