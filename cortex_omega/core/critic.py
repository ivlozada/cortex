from dataclasses import dataclass
from typing import List, Dict, Optional, Any
import math

from .rules import RuleBase, Scene, Rule, Literal, FactBase
from .config import KernelConfig
from .inference import InferenceEngine, infer
import logging

logger = logging.getLogger(__name__)

@dataclass
class Critic:
    config: KernelConfig

    def score_harmony(
        self,
        theory: RuleBase,
        scenes: List[Scene],
        entropy_map: Dict[str, float] = None,
    ) -> float:
        return self._score_harmony_logic(theory, scenes, entropy_map)

    def score_mdl(self, rule: Rule) -> float:
        return self._score_mdl_logic(rule)

    def _score_harmony_logic(
        self,
        theory: RuleBase,
        scenes: List[Scene],
        entropy_map: Dict[str, float] = None,
    ) -> float:
        score = evaluate_f1(theory, scenes)
        complexity = self._compute_complexity(theory, entropy_map)
        harmony = score / (1.0 + self.config.lambda_complexity * complexity)
        return harmony

    def _compute_complexity(self, theory: RuleBase, entropy_map: Dict[str, float] = None) -> float:
        num_rules = len(theory.rules)
        num_concepts = sum(1 for r in theory.rules.values() if "Concept_" in str(r.id))
        
        literal_cost = 0.0
        alpha = self.config.complexity_alpha
        base_cost = self.config.complexity_base_cost
        
        for r in theory.rules.values():
            if not r.is_safe:
                literal_cost += 1000.0 # Massive penalty for unsafe rules
            for lit in r.body:
                entropy_cost = 0.0
                if entropy_map and lit.predicate in entropy_map:
                    entropy_cost = alpha * entropy_map[lit.predicate]
                
                literal_cost += (base_cost + entropy_cost)

        return 1.0 * num_rules + literal_cost + self.config.complexity_concept_cost * num_concepts

    def _score_mdl_logic(self, rule: Rule) -> float:
        """
        Logic for MDL scoring.
        """
        if not rule.is_safe:
            return -1000.0 # Penalize unsafe rules heavily
            
        if rule.coverage == 0:
            return 0.0
        
        score = (rule.fires_pos - rule.fires_neg) - (self.config.lambda_complexity * rule.complexity)
        # print(f"DEBUG: Rule {rule.id} Score: {score:.4f} (TP={rule.fires_pos} TN={rule.fires_neg} C={rule.complexity})")
        return score





def clone_factbase(fb: FactBase) -> FactBase:
    """
    Copia profunda de un FactBase.
    """
    fb_copy = FactBase()
    for pred, args_set in fb.facts.items():
        for args in args_set:
            fb_copy.add(pred, args)
    return fb_copy

def evaluate_f1(theory: RuleBase, scenes: List[Scene]) -> float:
    """
    Calcula F1-Score de la teoría sobre un conjunto de escenas.
    """
    tp = 0
    fp = 0
    fn = 0
    
    for s in scenes:
        # CORTEX-OMEGA: Mask target predicate to prevent cheating
        # We must clone facts and remove the target label
        masked_facts = clone_factbase(s.facts)
        if s.target_predicate in masked_facts.facts:
            del masked_facts.facts[s.target_predicate]
        if f"NOT_{s.target_predicate}" in masked_facts.facts:
            del masked_facts.facts[f"NOT_{s.target_predicate}"]
            
        # Create temporary scene with masked facts
        masked_scene = Scene(
            id=s.id,
            facts=masked_facts,
            target_entity=s.target_entity,
            target_predicate=s.target_predicate,
            ground_truth=s.ground_truth,
            target_args=s.target_args
        )
        
        pred, _ = infer(theory, masked_scene)
        
        if pred and s.ground_truth:
            tp += 1
        elif pred and not s.ground_truth:
            fp += 1
        elif not pred and s.ground_truth:
            fn += 1
            
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    
    return 2 * (precision * recall) / (precision + recall + 1e-6)

def calculate_attribute_entropy(memory: List[Scene]) -> Dict[str, float]:
    """
    CORTEX-OMEGA: Calculate Shannon Entropy for each predicate in memory.
    H(A) = - Sum(p(x) * log2(p(x)))
    High Entropy = High Variance = Noise.
    Low Entropy = Stable = Signal.
    """
    import math
    
    # 1. Collect value distributions
    # pred -> {value -> count}
    distributions: Dict[str, Dict[str, int]] = {}
    
    for scene in memory:
        for pred, args_set in scene.facts.facts.items():
            for args in args_set:
                # Assuming unary or binary where 2nd arg is value
                val = "True"
                if len(args) == 2:
                    val = args[1]
                elif len(args) == 1:
                    val = "True"
                    
                if pred not in distributions:
                    distributions[pred] = {}
                
                if val not in distributions[pred]:
                    distributions[pred][val] = 0
                distributions[pred][val] += 1
                
    # 2. Calculate Entropy
    entropy_map = {}
    total_scenes = len(memory) if memory else 1
    
    for pred, counts in distributions.items():
        total_occurrences = sum(counts.values())
        # CORTEX-OMEGA: Min Support for Entropy Stability
        # If a predicate appears very few times, its entropy is unreliable (often 0.0).
        # We treat rare predicates as "High Entropy" (Unknown/Noise) by default.
        if total_occurrences < 3:
            entropy_map[pred] = 2.0 # High default penalty for rare attributes
            continue
            
        entropy = 0.0
        for val, count in counts.items():
            p = count / total_occurrences
            entropy -= p * math.log2(p)
            
        entropy_map[pred] = entropy
        # logger.debug(f"Entropy({pred}) = {entropy:.2f}")
        
    return entropy_map

def garbage_collect(theory: RuleBase, threshold: float = 0.0, config: Optional[KernelConfig] = None):
    """
    CORTEX-OMEGA Pillar 4: Concept Compression.
    Prunes low-utility rules using MDL Scoring.
    """
    to_remove: List[str] = [] # RuleID is str compatible mostly, but let's be careful
    # CORTEX-OMEGA v1.4: Use MDL Score
    # Threshold 0.0 means "Does more harm than good" (fires_neg > fires_pos - complexity penalty)
    
    lambda_val = config.lambda_complexity if config else 0.2
    
    for rid, rule in list(theory.rules.items()):
        # Give new rules a grace period (support_count < 5)
        if rule.support_count < 5:
            continue
            
        critic = Critic(config or KernelConfig())
        score = critic.score_mdl(rule)
        
        # If score is negative, the rule is actively harmful or too complex for its value
        if score < threshold:
            to_remove.append(rid)
    
    for rid in to_remove:
        theory.remove(rid)

    if to_remove:
        logger.info(f"CORTEX: Garbage Collector pruned {len(to_remove)} rules (MDL < {threshold}): {to_remove}")

    # CORTEX-OMEGA v1.4: Redundancy Pruning (Compression)
    prune_redundant_rules(theory)


def prune_redundant_rules(theory: RuleBase):
    """
    Removes rules that are subsumed by more general rules with equal or better reliability.
    """
    to_remove = set()
    rules = list(theory.rules.values())
    
    for i in range(len(rules)):
        for j in range(len(rules)):
            if i == j: continue
            
            r_gen = rules[i]
            r_spec = rules[j]
            
            if r_spec.id in to_remove: continue
            if r_gen.id in to_remove: continue
            
            # Check if r_gen subsumes r_spec
            if r_spec.is_subsumed_by(r_gen):
                # r_gen is more general (or equal)
                
                # Check reliability
                # If general rule is reliable enough, we don't need the specific one
                if r_gen.reliability >= r_spec.reliability:
                    to_remove.add(r_spec.id)
                elif r_gen.reliability > 0.9:
                    # Even if specific is 1.0 and general is 0.95, general is preferred
                    to_remove.add(r_spec.id)
                    
    for rid in to_remove:
        theory.remove(rid)
        
    if to_remove:
        logger.info(f"CORTEX: Compressed {len(to_remove)} redundant rules: {to_remove}")

