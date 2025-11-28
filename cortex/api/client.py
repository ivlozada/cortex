"""
Cortex API: The God Class
=========================
High-level interface for the Cortex Neuro-Symbolic Engine.
Provides the "5-Line Experience".
"""

from typing import Dict, Any, List, Optional
from ..core.engine import update_theory_kernel, KernelConfig, infer
from ..core.rules import RuleBase, FactBase, Literal, Rule
from ..core.values import ValueBase, Axiom
from ..core.hypothesis import HypothesisGenerator
from ..io.ingestor import SmartIngestor
import time

class Cortex:
    """
    The main entry point for the Cortex Epistemic Engine.

    This class orchestrates the interaction between the Logic Engine, 
    Memory Systems, and Hypothesis Generators. It provides a high-level 
    API for ingesting data and querying logical truths.

    Attributes:
        config (KernelConfig): Configuration parameters for the engine.
        theory (RuleBase): The current set of crystallized logical rules.
        axioms (ValueBase): The repository of established truths.
    """
    
    def __init__(self, sensitivity: float = 0.1):
        """
        Initializes the Cortex Brain.

        Args:
            sensitivity (float): The lambda complexity penalty. 
                                 Higher values make the brain more skeptical of noise.
                                 Defaults to 0.1.
        """
        self.config = KernelConfig(lambda_complexity=sensitivity)
        self.theory = RuleBase()
        self.memory = []
        self.axioms = ValueBase()
        self.ingestor = SmartIngestor()
        
        # Initialize sub-components
        if not self.config.patch_generator:
            self.config.patch_generator = HypothesisGenerator()
            
    def absorb(self, source: str):
        """
        Ingests raw data from a file, normalizes it, and updates the logic model.

        This method automatically detects file format (CSV/JSON), applies 
        entropy regularization to filter noise, and performs structural 
        gradient updates on the internal theory.

        Args:
            source (str): Path to the source file (e.g., 'data.csv').

        Returns:
            None
        
        Raises:
            FileNotFoundError: If the file path is invalid.
        """
        print(f"🧠 Cortex is absorbing knowledge from '{source}'...")
        scenes = self.ingestor.ingest(source)
        
        start_time = time.time()
        for i, scene in enumerate(scenes):
            if i % 10 == 0:
                print(f"  Processing datum {i}/{len(scenes)}...")
            self.theory, self.memory = update_theory_kernel(
                self.theory, scene, self.memory, self.axioms, self.config
            )
        
        duration = time.time() - start_time
        print(f"✨ Enlightenment achieved in {duration:.2f}s.")
        print(f"📚 Learned {len(self.theory.rules)} rules.")
        
    def absorb_memory(self, data: List[Dict[str, Any]], target_label: str):
        """
        Programmatic ingestion of data (list of dicts).
        """
        from ..core.rules import Scene, FactBase
        
        for i, item in enumerate(data):
            # Generate ID
            scene_id = f"mem_{len(self.memory)}_{i}"
            
            # Extract ground truth
            if target_label in item:
                ground_truth = item[target_label]
                exclude_keys = {target_label, "result"}
            else:
                ground_truth = item.get("result", False)
                exclude_keys = {"result"}
            
            # Build facts
            facts = FactBase()
            target_entity = "obj"
            
            for key, val in item.items():
                if key in exclude_keys: continue
                facts.add(key, (target_entity, val))
                
            # Create Scene
            scene = Scene(
                id=scene_id,
                facts=facts,
                target_entity=target_entity,
                target_predicate=target_label,
                ground_truth=ground_truth
            )
            
            # Learn
            self.theory, self.memory = update_theory_kernel(
                self.theory, scene, self.memory, self.axioms, self.config
            )

    def query(self, **kwargs) -> 'InferenceResult':
        """
        Queries the engine with a set of observations.
        Example: brain.query(mass="heavy", type="guest", target="fraud")
        """
        # 1. Construct a temporary scene/facts from kwargs
        facts = FactBase()
        entity = "query_entity"
        target_pred = kwargs.get("target") # None by default
        
        for pred, val in kwargs.items():
            if pred == "target": continue
            facts.add(pred, (entity, val))
            
        # 2. Run inference
        from ..core.inference import InferenceEngine
        engine = InferenceEngine(facts, self.theory)
        derived = engine.forward_chain()
        
        # Determine prediction and explanation
        prediction = False
        explanation = "No rule fired."
        confidence = 0.0
        
        # If target is explicit, check it
        if target_pred:
            target_lit = Literal(target_pred, (entity,))
            prediction = facts.contains(target_lit)
            
            # Find explanation
            trace = engine.trace
            for step in reversed(trace):
                if step["type"] == "derivation" and step["derived"].predicate == target_pred:
                    rule_id = step["rule_id"]
                    rule = self.theory.rules.get(rule_id)
                    if rule:
                        explanation = str(rule)
                        confidence = rule.confidence
                        break
        else:
            # Implicit target: Look for ANY derivation
            trace = engine.trace
            if trace:
                # Find the last derivation
                for step in reversed(trace):
                    if step["type"] == "derivation":
                        # Found a rule that fired!
                        rule_id = step["rule_id"]
                        rule = self.theory.rules.get(rule_id)
                        if rule:
                            derived = step.get("derived")
                            if derived and derived.negated:
                                prediction = False
                            else:
                                prediction = True
                                
                            explanation = str(rule)
                            confidence = rule.confidence
                            # We assume the first one we find (last in trace) is the "answer"
                            break
                            
        # 3. Explicit Negation Check
        if not prediction and target_pred:
            # Check if we derived NOT_target
            neg_target_lit = Literal(f"NOT_{target_pred}", (entity,))
            if facts.contains(neg_target_lit):
                # We have an explicit negative derivation!
                # Find explanation in trace
                for step in reversed(engine.trace):
                    if step["type"] == "derivation" and step["derived"].predicate == f"NOT_{target_pred}":
                        rule_id = step["rule_id"]
                        rule = self.theory.rules.get(rule_id)
                        if rule:
                            explanation = str(rule)
                            # We keep prediction=False, but provide explanation
                            break
                            
        return InferenceResult(prediction, explanation, confidence)

    def context_shift(self, context_name: str):
        """
        Simulates a context shift (Plasticity).
        Degrades current rules to allow new learning.
        """
        print(f"🔄 Context Shift: {context_name}")
        # Simple implementation: Decay all rule confidence
        for rule in self.theory.rules.values():
            rule.confidence *= 0.5
            
    def add_exception(self, rule: str, condition: str):
        """
        Manually adds an exception.
        Placeholder for advanced logic.
        """
        print(f"🛡️ Exception added: {rule} IF {condition}")
        # TODO: Parse and add to RuleBase/ValueBase

class InferenceResult:
    def __init__(self, prediction: bool, axiom: str, confidence: float):
        self.prediction = prediction
        self.axiom = axiom
        self.confidence = confidence
        
    def __repr__(self):
        return f"Result(pred={self.prediction}, conf={self.confidence:.2f})"
