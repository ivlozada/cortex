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
from ..core.inference import Proof
from ..core.errors import DataFormatError, RuleParseError, CortexError

from ..io.ingestor import SmartIngestor
import time
import logging

# Configure default logging to WARNING to keep the launch clean
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
        self.facts = FactBase() # CORTEX-OMEGA: Persistent Knowledge Base
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
            DataFormatError: If the file format is not supported.
        """
        logger.info(f"🧠 Cortex is absorbing knowledge from '{source}'...")
        try:
            scenes = self.ingestor.ingest(source)
        except Exception as e:
            if isinstance(e, FileNotFoundError):
                raise
            raise DataFormatError(f"Failed to ingest data from {source}: {str(e)}") from e
        
        start_time = time.time()
        for i, scene in enumerate(scenes):
            if i % 10 == 0:
                logger.debug(f"  Processing datum {i}/{len(scenes)}...")
            self.theory, self.memory = update_theory_kernel(
                self.theory, scene, self.memory, self.axioms, self.config
            )
        
        duration = time.time() - start_time
        logger.info(f"✨ Enlightenment achieved in {duration:.2f}s.")
        logger.info(f"📚 Learned {len(self.theory.rules)} rules.")

        
    def _sanitize_value(self, val):
        """
        Normalizes values to strings for symbolic processing.
        """
        if isinstance(val, bool):
            return str(val).lower()
        return str(val)

    def absorb_memory(self, data: List[Dict[str, Any]], target_label: str):
        """
        Programmatic ingestion of data (list of dicts).
        """
        from ..core.rules import Scene, FactBase
        
        for i, item in enumerate(data):
            # Generate ID
            scene_id = f"mem_{len(self.memory)}_{i}"
            
            # Extract ground truth
            # Extract ground truth
            if target_label in item:
                ground_truth = item[target_label]
                exclude_keys = {target_label, "result"}
            elif f"is_{target_label}" in item:
                ground_truth = item[f"is_{target_label}"]
                exclude_keys = {f"is_{target_label}", "result"}
            else:
                ground_truth = item.get("result", False)
                exclude_keys = {"result"}
            
            # Build facts
            facts = FactBase()
            target_entity = item.get("id", "obj") # Use ID if available
            
            for key, val in item.items():
                if key == "id": continue
                if key in exclude_keys: continue
                val = self._sanitize_value(val)
                
                # Standard Fact: predicate(entity, value)
                facts.add(key, (target_entity, val))
                self.facts.add(key, (target_entity, val))
                
                # CORTEX-OMEGA: Smart Boolean Handling
                if val == "true":
                    # Add arity-1 fact: predicate(entity)
                    facts.add(key, (target_entity,))
                    self.facts.add(key, (target_entity,))
                    
                    # Handle is_ prefix
                    if key.startswith("is_"):
                        stripped = key[3:]
                        facts.add(stripped, (target_entity,))
                        self.facts.add(stripped, (target_entity,))
                
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
        print("DEBUG: ENTERING QUERY")
        # 1. Construct a temporary scene/facts from kwargs
        # Start with persistent facts
        import copy
        facts = copy.deepcopy(self.facts)
        
        entity = kwargs.get("id", "query_entity") # Use ID if provided
        target_pred = kwargs.get("target") # None by default
        
        for key, val in kwargs.items():
            if key == "target": continue
            if key == "id": continue # Don't add ID as a property of itself
            val = self._sanitize_value(val)
            
            # Standard Fact: predicate(entity, value)
            facts.add(key, (entity, val))
            
            # Smart Boolean Handling (Consistency with absorb_memory)
            if val == "true":
                # Add arity-1 fact: predicate(entity)
                facts.add(key, (entity,))
                
                # Handle is_ prefix
                if key.startswith("is_"):
                    stripped = key[3:]
                    facts.add(stripped, (entity,))
            
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
                            
                            
        # 4. Reconstruct Proof (Traceback)
        proof = None
        if prediction and target_pred:
            target_lit = Literal(target_pred, (entity,))
            proof = engine.get_proof(target_lit)
        elif not prediction and target_pred:
             # Check for explicit negation proof
             neg_target_lit = Literal(f"NOT_{target_pred}", (entity,))
             if facts.contains(neg_target_lit):
                 proof = engine.get_proof(neg_target_lit)

        return InferenceResult(prediction, explanation, confidence, proof)


    def context_shift(self, context_name: str):
        """
        Simulates a context shift (Plasticity).
        Degrades current rules to allow new learning.
        """
        logger.info(f"🔄 Context Shift: {context_name}")
        # Simple implementation: Decay all rule confidence

        for rule in self.theory.rules.values():
            rule.confidence *= 0.5
            
    def add_rule(self, rule_str: str):
        """
        Manually adds a rule to the theory.
        Example: brain.add_rule("fraud(X) :- transaction(X, amount, V), V > 10000")
        
        Raises:
            RuleParseError: If the rule string is invalid.
        """
        from ..core.rules import parse_rule
        try:
            rule = parse_rule(rule_str)
        except Exception as e:
            raise RuleParseError(f"Invalid rule string '{rule_str}': {str(e)}") from e
            
        self.theory.add(rule)
        logger.info(f"🔧 Rule added: {rule}")


    def add_exception(self, rule: str, condition: str):
        """
        Manually adds an exception.
        Placeholder for advanced logic.
        """
        logger.info(f"🛡️ Exception added: {rule} IF {condition}")
        # TODO: Parse and add to RuleBase/ValueBase


class InferenceResult:
    def __init__(self, prediction: bool, explanation: str, confidence: float, proof: Optional[Proof] = None):
        self.prediction = prediction
        self.explanation = explanation
        self.confidence = confidence
        self.proof = proof

        
    def __repr__(self):
        return f"Result(pred={self.prediction}, conf={self.confidence:.2f})"
