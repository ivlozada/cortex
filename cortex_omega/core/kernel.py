"""
Cortex-Omega Kernel (v1.0)
==========================
The Sacred Core.
This module defines the public API of the GDM Kernel.
"""

import logging
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import asdict

from .rules import Scene, RuleBase, FactBase
from .config import KernelConfig
from .learner import Learner
from .engine import infer
from .values import ValueBase

logger = logging.getLogger(__name__)

class GDMKernel:
    """
    The Grandmaster (GDM) Kernel.
    
    Mission:
    "Given a set of scenes (examples) and a target predicate, produce a logical theory
    (set of rules) that explains the target with maximum accuracy and minimum complexity."
    """
    
    def __init__(self, config: Optional[KernelConfig] = None):
        self.config = config or KernelConfig()
        self.theory: Optional[RuleBase] = None
        self.memory: List[Scene] = []
        self.axioms = ValueBase()
        self.learner = Learner(self.config)
        
    def fit(self, scenes: List[Scene], target_predicate: str, axioms: Optional[ValueBase] = None) -> 'GDMKernel':
        """
        Trains the kernel on the provided scenes.
        
        Args:
            scenes: List of training examples.
            target_predicate: The predicate to learn.
            axioms: Optional axiomatic constraints.
            
        Returns:
            self: For chaining.
        """
        if not scenes:
            logger.warning("GDMKernel.fit called with empty scenes list.")
            return self
            
        assert all(isinstance(s, Scene) for s in scenes), "All items in scenes must be Scene objects."
        assert isinstance(target_predicate, str) and target_predicate, "Target predicate must be a non-empty string."
            
        if axioms:
            self.axioms = axioms
            
        # Initialize theory if empty
        if self.theory is None:
            self.theory = RuleBase()
            
        logger.info(f"GDMKernel: Starting training on {len(scenes)} scenes for target '{target_predicate}'.")
        
        # Main Learning Loop
        for i, scene in enumerate(scenes):
            # Ensure scene has correct target
            if scene.target_predicate != target_predicate:
                # In strict mode, this might be an error. For now, warn.
                logger.warning(f"Scene {scene.id} has target '{scene.target_predicate}', expected '{target_predicate}'. Skipping.")
                continue
                
            # Update theory using Learner
            self.theory, self.memory = self.learner.learn(
                self.theory,
                scene,
                self.memory,
                self.axioms
            )
            
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{len(scenes)} scenes. Rules: {len(self.theory.rules)}")
                
        logger.info("GDMKernel: Training complete.")
        return self

    def predict(self, scenes: List[Scene]) -> List[bool]:
        """
        Infers the target predicate for a list of scenes.
        
        Args:
            scenes: List of scenes to predict.
            
        Returns:
            List[bool]: Predictions (True/False).
        """
        if self.theory is None:
            logger.warning("GDMKernel.predict called before fit(). Returning all False.")
            return [False] * len(scenes)
            
        predictions = []
        for scene in scenes:
            pred, _ = infer(self.theory, scene, self.config)
            predictions.append(pred)
            
        return predictions

    def explain(self) -> Dict[str, Any]:
        """
        Returns the internal theory in a structured format.
        
        Returns:
            Dict: Structured representation of the theory and metrics.
        """
        if self.theory is None:
            return {"status": "uninitialized", "rules": []}
            
        rules_data = [r.to_dict() for r in self.theory.rules.values()]
        
        # Calculate basic metrics on memory (if available)
        metrics: Dict[str, float] = {}
        if self.memory:
            # Re-evaluate on memory to get current metrics
            # This might be expensive, so maybe just return stored stats?
            # For now, let's return rule stats.
            pass
            
        return {
            "status": "trained",
            "rule_count": len(self.theory.rules),
            "rules": rules_data,
            "config": asdict(self.config)
        }
