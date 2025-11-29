import unittest
from cortex_omega import Cortex

class TestLogic(unittest.TestCase):
    def setUp(self):
        self.brain = Cortex()

    def test_missing_link(self):
        """
        Test that a missing link in the chain prevents derivation.
        """
        self.brain.add_rule("mortal(X) :- human(X)")
        # Missing: human(X) :- greek(X)
        
        # Fact: socrates is greek
        # But we don't know greek -> human
        res = self.brain.query(id="socrates", greek=True, target="mortal")
        self.assertFalse(res.prediction, "Should not predict mortal without the bridge rule")

    def test_cycle_detection(self):
        """
        Test that the engine handles cycles gracefully (doesn't hang).
        """
        # Cycle: A :- B, B :- A
        self.brain.add_rule("A(X) :- B(X)")
        self.brain.add_rule("B(X) :- A(X)")
        
        # Inject A
        # Query B
        # Ideally, it should derive B from A in one step and stop.
        # If we query C where C :- B, it should also work.
        
        # If we have no facts, it shouldn't infinite loop.
        res = self.brain.query(id="test", target="A")
        self.assertFalse(res.prediction)
        
        # If we have A, we should get B.
        res_b = self.brain.query(id="test", A=True, target="B")
        self.assertTrue(res_b.prediction)

if __name__ == '__main__':
    unittest.main()
