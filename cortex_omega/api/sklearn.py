
"""
Cortex Scikit-Learn Wrapper
===========================
Provides a standard sklearn-compatible interface for Cortex Omega.
Allows usage in Pipelines, GridSearchCV, etc.
"""

from typing import List, Dict, Any, Union
import pandas as pd
import numpy as np

try:
    from sklearn.base import BaseEstimator, ClassifierMixin
except ImportError:
    # Fallback if sklearn is not installed
    class BaseEstimator: pass
    class ClassifierMixin: pass

from .client import Cortex

class CortexClassifier(BaseEstimator, ClassifierMixin):
    """
    A Scikit-Learn compatible wrapper for Cortex Omega.
    
    Parameters:
    -----------
    target_label : str
        The name of the target column/predicate to learn.
    sensitivity : float, default=0.1
        The lambda complexity penalty.
    mode : str, default="robust"
        "strict" or "robust".
    """
    
    def __init__(self, target_label: str, sensitivity: float = 0.1, mode: str = "robust"):
        self.target_label = target_label
        self.sensitivity = sensitivity
        self.mode = mode
        self.brain = None
        self.classes_ = [False, True] # Binary classifier by default
        
    def fit(self, X: Union[pd.DataFrame, List[Dict]], y=None):
        """
        Fit the Cortex model.
        
        Parameters:
        -----------
        X : DataFrame or List of Dicts
            Training data.
        y : array-like, optional
            Target values. If provided, they are merged into X.
            If X is a DataFrame/List containing the target, y can be None.
        """
        self.brain = Cortex(sensitivity=self.sensitivity, mode=self.mode)
        
        data = self._convert_to_dicts(X, y)
        
        # Absorb memory
        self.brain.absorb_memory(data, target_label=self.target_label)
        
        return self
        
    def predict(self, X: Union[pd.DataFrame, List[Dict]]) -> np.ndarray:
        """
        Predict class labels for X.
        """
        if not self.brain:
            raise RuntimeError("Model not fitted yet.")
            
        data = self._convert_to_dicts(X)
        predictions = []
        
        for item in data:
            # Query the brain
            # We pass the item as kwargs
            result = self.brain.query(target=self.target_label, **item)
            predictions.append(result.prediction)
            
        return np.array(predictions)
        
    def predict_proba(self, X: Union[pd.DataFrame, List[Dict]]) -> np.ndarray:
        """
        Predict class probabilities for X.
        Returns (n_samples, 2) array.
        """
        if not self.brain:
            raise RuntimeError("Model not fitted yet.")
            
        data = self._convert_to_dicts(X)
        probas = []
        
        for item in data:
            result = self.brain.query(target=self.target_label, **item)
            # Confidence is for the predicted class
            p = result.confidence
            if result.prediction:
                probas.append([1.0 - p, p])
            else:
                probas.append([p, 1.0 - p])
                
        return np.array(probas)
        
    def _convert_to_dicts(self, X, y=None) -> List[Dict]:
        """Helper to convert input to list of dicts."""
        if isinstance(X, pd.DataFrame):
            data = X.to_dict(orient="records")
        elif isinstance(X, list):
            data = X
        else:
            raise ValueError("X must be a pandas DataFrame or a list of dicts.")
            
        if y is not None:
            # Merge y into data
            y_list = list(y)
            if len(y_list) != len(data):
                raise ValueError("X and y must have same length.")
                
            for i, item in enumerate(data):
                # We modify the dict in place (assuming it's safe for internal use)
                # Or copy it
                item[self.target_label] = y_list[i]
                
        return data
