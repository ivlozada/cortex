from dataclasses import dataclass
from typing import List, Dict, Optional, Any
import math

from .rules import RuleBase, Scene, Rule, Literal, FactBase
from .config import KernelConfig
from .inference import InferenceEngine

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
        return score / (1.0 + self.config.lambda_complexity * complexity)

    def _compute_complexity(self, theory: RuleBase, entropy_map: Dict[str, float] = None) -> float:
        num_rules = len(theory.rules)
        num_concepts = sum(1 for r in theory.rules.values() if "Concept_" in str(r.id))
        
        literal_cost = 0.0
        alpha = self.config.complexity_alpha
        base_cost = self.config.complexity_base_cost
        
        for r in theory.rules.values():
            for lit in r.body:
                entropy_cost = 0.0
                if entropy_map and lit.predicate in entropy_map:
                    entropy_cost = alpha * entropy_map[lit.predicate]
                
                literal_cost += (base_cost + entropy_cost)

        return 1.0 * num_rules + literal_cost + self.config.complexity_concept_cost * num_concepts

    def _score_mdl_logic(self, rule: Rule) -> float:
        if rule.coverage == 0:
            return 0.0
        return (rule.fires_pos - rule.fires_neg) - (self.config.lambda_complexity * rule.complexity)





def clone_factbase(fb: FactBase) -> FactBase:
    """
    Copia profunda de un FactBase.
    """
    fb_copy = FactBase()
    for pred, args_set in fb.facts.items():
        for args in args_set:
            fb_copy.add(pred, args)
    return fb_copy
