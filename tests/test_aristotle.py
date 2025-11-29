import unittest
from cortex_omega.api.client import Cortex

class TestAristotle(unittest.TestCase):
    def test_socrates_mortality(self):
        print(f"--- 🏛️  Running The Aristotle Test ---")
        
        # 1. Initialize
        engine = Cortex()

        # 2. Inject Rule (The 'Brain' Test)
        # "X is mortal IF X is a man"
        print("Step 1: Teaching Logic (All men are mortal)...")
        engine.add_rule("mortal(X) :- man(X)")

        # 3. Inject Data (The 'Memory' Test)
        # "Socrates is a man"
        print("Step 2: Injecting Data (Socrates is a man)...")
        socrates_data = [{"id": "socrates", "is_man": True}]
        # We map the JSON key 'is_man' to the predicate 'man'
        # Use a dummy target so 'is_man' is treated as a fact, not the label to be predicted
        engine.absorb_memory(socrates_data, target_label="dummy")

        # 4. Query (The 'Inference' Test)
        print("Step 3: Asking Question (Is Socrates mortal?)...")
        result = engine.query(id="socrates", target="mortal")

        self.assertTrue(result.prediction, "Socrates should be mortal")
        print(f"\n✅ PASSED: Socrates is confirmed mortal. (Confidence: {result.confidence})")

if __name__ == '__main__':
    unittest.main()
