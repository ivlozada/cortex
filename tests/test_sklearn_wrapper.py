
import unittest
from cortex_omega.api.sklearn import CortexClassifier

class TestSklearnWrapper(unittest.TestCase):
    def test_fit_predict(self):
        """
        Verify fit and predict workflow.
        """
        # Training data (XOR-like but simpler: heavy sinks)
        X_train = [
            {"material": "wood", "density": "low"},
            {"material": "wood", "density": "low"},
            {"material": "iron", "density": "high"},
            {"material": "iron", "density": "high"},
        ]
        y_train = [False, False, True, True] # Target: sinks
        
        clf = CortexClassifier(target_label="sinks")
        clf.fit(X_train, y_train)
        
        # Test data
        X_test = [
            {"material": "wood", "density": "low"},
            {"material": "iron", "density": "high"},
        ]
        
        preds = clf.predict(X_test)
        
        self.assertEqual(len(preds), 2)
        self.assertFalse(preds[0]) # Wood -> False
        self.assertTrue(preds[1])  # Iron -> True
        
    def test_predict_proba(self):
        """
        Verify predict_proba returns probabilities.
        """
        X_train = [
            {"feature": "A", "val": "1"},
            {"feature": "A", "val": "1"},
        ]
        y_train = [True, True]
        
        clf = CortexClassifier(target_label="target")
        clf.fit(X_train, y_train)
        
        X_test = [{"feature": "A", "val": "1"}]
        probas = clf.predict_proba(X_test)
        
        self.assertEqual(probas.shape, (1, 2))
        self.assertGreater(probas[0][1], 0.5) # Prob of True should be high

if __name__ == '__main__':
    unittest.main()
