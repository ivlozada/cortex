"""
Cortex API: The God Class
=========================
High-level interface for the Cortex Neuro-Symbolic Engine.
Provides the "5-Line Experience".
"""

import logging
import pickle
from typing import List, Dict, Optional, Any, Union
from ..core.learner import Learner
from ..core.config import KernelConfig
from ..core.inference import infer, Proof
from ..core.rules import RuleBase, FactBase, Literal, Rule
from ..core.values import ValueBase, Axiom
from ..io.ingestor import SmartIngestor
from ..core.hypothesis import HypothesisGenerator

from ..core.errors import DataFormatError, RuleParseError, CortexError, EpistemicVoidError # Added EpistemicVoidError

import time
import json # Added for export_rules

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
    
    def __init__(self, 
                 sensitivity: float = 0.1, 
                 mode: str = "robust",
                 priors: Dict[str, float] = None,
                 noise_model: Dict[str, float] = None,
                 plasticity: Dict[str, Any] = None,
                 feature_priors: Dict[str, float] = None):
        """
        Initializes the Cortex Brain.

        Args:
            sensitivity (float): The lambda complexity penalty. 
                                 Higher values make the brain more skeptical of noise.
                                 Defaults to 0.1.
            mode (str): "robust" (default) or "strict".
                        "robust": Resilient to noise, requires multiple counter-examples to override.
                        "strict": Logical purity, single counter-example overrides immediately.
            priors (dict): Bayesian priors for rule generation. e.g. {"rule_base": 0.5, "exception": 0.3}
            noise_model (dict): Expected noise rates. e.g. {"false_positive": 0.05, "false_negative": 0.05}
            plasticity (dict): Parameters for rule retention. e.g. {"min_conf_to_keep": 0.6, "max_rule_count": 500}
            feature_priors (dict): Causal hints for feature selection. e.g. {"is_heavy": 0.9, "color": 0.1}
        """
        self.config = KernelConfig(
            lambda_complexity=sensitivity, 
            mode=mode,
            priors=priors or {"rule_base": 0.5, "exception": 0.3},
            noise_model=noise_model or {"false_positive": 0.05, "false_negative": 0.05},
            plasticity=plasticity or {"min_conf_to_keep": 0.6, "max_rule_count": 500},
            feature_priors=feature_priors or {}
        )
        self.theory = RuleBase()
        self.axioms = ValueBase()
        self.memory: List[Any] = [] # List of Scenes
        self.facts = FactBase() # CORTEX-OMEGA: Persistent Knowledge Base
        
        # Initialize Learner
        self.learner = Learner(self.config)
        
        # Initialize Ingestor
        self.ingestor = SmartIngestor()
        
        # Initialize sub-components
        if not self.config.patch_generator:
            self.config.patch_generator = HypothesisGenerator(config=self.config)
            
    def set_mode(self, mode: str):
        """
        Dynamically switches the operating mode.
        Args:
            mode (str): "strict" or "robust".
        """
        mode = mode.lower()
        if mode not in ["strict", "robust"]:
            raise ValueError("Mode must be 'strict' or 'robust'")
        self.config.mode = mode
        logger.info(f"🔄 Switched to {mode.upper()} mode.")
            
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
            self.theory, self.memory = self.learner.learn(
                self.theory, scene, self.memory, self.axioms
            )
        
        duration = time.time() - start_time
        logger.info(f"✨ Enlightenment achieved in {duration:.2f}s.")
        logger.info(f"📚 Learned {len(self.theory.rules)} rules.")

        
    def _sanitize_value(self, val):
        """
        Normalizes values to strings for symbolic processing, but preserves numbers.
        """
        if isinstance(val, bool):
            return str(val).lower()
        if isinstance(val, (int, float)):
            return val
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
                
            # Normalize Boolean Strings
            if isinstance(ground_truth, str):
                if ground_truth.lower() == "true":
                    ground_truth = True
                elif ground_truth.lower() == "false":
                    ground_truth = False
            
            # Build facts
            
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
            
            # CORTEX-OMEGA: Persist the target as a fact if it is True
            # This allows subsequent queries to use this learned/observed truth.
            if ground_truth is True:
                self.facts.add(target_label, (target_entity,))
            
            # Learn
            # Learn
            self.theory, self.memory = self.learner.learn(
                self.theory, scene, self.memory, self.axioms
            )

    def query(self, **kwargs) -> 'InferenceResult':
        """
        Queries the engine with a set of observations.
        Example: brain.query(mass="heavy", type="guest", target="fraud")
        """
        from ..core.rules import Scene, FactBase, Literal
        import copy

        entity = kwargs.get("id", "query_entity")
        target_pred = kwargs.get("target")

        # === Epistemic Void guard ===
        # If there's no theory AND no fact about target, bail out explicitly.
        if not self.theory.rules and target_pred:
            # Check if we *already know* target(entity) as a fact
            direct_fact = Literal(target_pred, (entity,))
            if not self.facts.contains(direct_fact):
                raise EpistemicVoidError(
                    f"No rules or facts available for target '{target_pred}'. "
                    "The Cortex has no epistemic basis to answer this query."
                )

        # 1. Start with persistent facts
        facts = copy.deepcopy(self.facts)
        from ..core.rules import Scene, FactBase
        import copy
        
        # Start with persistent facts
        facts = copy.deepcopy(self.facts)
        
        entity = kwargs.get("id", "query_entity")
        target_pred = kwargs.get("target")
        
        for key, val in kwargs.items():
            if key == "target": continue
            if key == "id": continue
            val = self._sanitize_value(val)
            
            # Standard Fact: predicate(entity, value)
            facts.add(key, (entity, val))
            
            # Smart Boolean Handling
            if val == "true":
                facts.add(key, (entity,))
                if key.startswith("is_"):
                    stripped = key[3:]
                    facts.add(stripped, (entity,))
            
        # Create a temporary Scene
        scene = Scene(
            id="query_scene",
            facts=facts,
            target_entity=entity,
            target_predicate=target_pred if target_pred else "unknown",
            ground_truth=False # Dummy
        )
        
        # 2. Run inference using the Kernel's infer function (which has Conflict Resolution)
        from ..core.engine import infer
        from ..core.inference import InferenceEngine, Literal
        
        prediction, trace = infer(self.theory, scene)
        
        # 3. Extract Explanation and Confidence
        explanation = "No rule fired."
        confidence = 0.0
        proof = None
        
        # If prediction is True, find the positive proof
        if prediction:
            # Re-run get_proof to get the object (infer returns bool, trace)
            # Ideally infer should return the proof object too, but for now we reconstruct
            
            # Let's just trust the trace for explanation
            for step in reversed(trace):
                if step["type"] == "derivation":
                    # Check if this derivation matches target
                    derived = step["derived"]
                    if target_pred and derived.predicate == target_pred:
                        rule_id = step["rule_id"]
                        rule = self.theory.rules.get(rule_id)
                        if rule:
                            explanation = str(rule)
                            confidence = rule.confidence
                            break
                    elif not target_pred:
                         # Implicit target
                         rule_id = step["rule_id"]
                         rule = self.theory.rules.get(rule_id)
                         if rule:
                            explanation = str(rule)
                            confidence = rule.confidence
                            break
                            
            # Reconstruct Proof Object
            if target_pred:
                # We need an engine instance to call get_proof
                engine = InferenceEngine(facts, self.theory)
                engine.forward_chain() # Re-run to populate provenance
                target_lit = Literal(target_pred, (entity,))
                proof = engine.get_proof(target_lit)
                
        else:
            # Prediction False. Check for Explicit Negation explanation
            # infer handles conflict resolution, so if it returned False, 
            # it might be because NOT_target won.
            
            if target_pred:
                neg_target = f"NOT_{target_pred}"
                # Check trace for negative derivation
                for step in reversed(trace):
                    if step["type"] == "derivation":
                        derived = step["derived"]
                        if derived.predicate == neg_target:
                            rule_id = step["rule_id"]
                            rule = self.theory.rules.get(rule_id)
                            if rule:
                                explanation = str(rule)
                                confidence = rule.confidence
                                break
                
                # Reconstruct Proof Object for Negation
                engine = InferenceEngine(facts, self.theory)
                engine.forward_chain()
                neg_lit = Literal(neg_target, (entity,))
                if facts.contains(neg_lit):
                    proof = engine.get_proof(neg_lit)

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

    def inspect_rules(self, target: str = None) -> List[Rule]:
        """
        Returns a list of rules, optionally filtered by target predicate.
        Useful for debugging and analysis.
        """
        if target:
            # Return rules that have 'target' in the head
            # Note: target might be a predicate name
            return [r for r in self.theory.rules.values() if r.head.predicate == target]
        return list(self.theory.rules.values())

    def export_rules(self, format: str = "json") -> str:
        """
        Exports the learned rules to a standard format.
        
        Args:
            format (str): "json" or "prolog".
            
        Returns:
            str: The exported rules.
        """
        rules = list(self.theory.rules.values())
        
        if format.lower() == "json":
            export_data = [r.to_dict() for r in rules]
            return json.dumps(export_data, indent=2)
            
        elif format.lower() == "prolog":
            lines = []
            for r in rules:
                # Convert to Prolog syntax: head :- body1, body2.
                head_str = str(r.head).replace("(", "(").replace(")", ")")
                if r.head.negated:
                    head_str = "not_" + head_str[1:] # Remove ¬
                
                body_parts = []
                for b in r.body:
                    b_str = str(b)
                    if b.negated:
                        b_str = "not_" + b_str[1:]
                    body_parts.append(b_str)
                
                body_str = ", ".join(body_parts)
                if body_str:
                    lines.append(f"{head_str} :- {body_str}.")
                else:
                    lines.append(f"{head_str}.")
                    
            return "\n".join(lines)
        else:
            raise ValueError(f"Unknown format: {format}")

    def save_brain(self, file_path: str):
        """
        Saves the entire brain state to a file.
        
        Args:
            file_path (str): Path to the output file.
        """
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)
        logger.info(f"Brain saved to {file_path}")

    @staticmethod
    def load_brain(file_path: str) -> 'Cortex':
        """
        Loads a brain state from a file.
        
        Args:
            file_path (str): Path to the input file.
            
        Returns:
            Cortex: The loaded brain instance.
        """
        with open(file_path, 'rb') as f:
            brain = pickle.load(f)
        logger.info(f"Brain loaded from {file_path}")
        return brain
class InferenceResult:
    def __init__(self, prediction: bool, explanation: str, confidence: float, proof: Optional[Proof] = None):
        self.prediction = prediction
        self.explanation = explanation
        self.confidence = confidence
        self.proof = proof

        
    def __repr__(self):
        return f"Result(pred={self.prediction}, conf={self.confidence:.2f})"
