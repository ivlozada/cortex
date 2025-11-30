# core_kernel.py

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any

from .rules import RuleBase, Rule, Literal, FactBase, Scene, RuleID
from .inference import InferenceEngine, infer
from .values import ValueBase, Axiom
from .hypothesis import HypothesisGenerator, FailureContext
from .config import KernelConfig
from .critic import Critic
from .learner import Learner

import copy
import logging

logger = logging.getLogger(__name__)

# =========================
# 1. Operador de actualización
# =========================




class KnowledgeCompiler:
    """
    CORTEX-OMEGA Pillar 5: Self-Reflecting Compiler.
    Promotes stable rules to axioms and detects logical conflicts.
    """
    def __init__(self, promotion_threshold: float = 0.99, min_usage: int = 10):
        self.promotion_threshold = promotion_threshold
        self.min_usage = min_usage

    def promote_stable_rules(self, theory: RuleBase, axioms: ValueBase) -> List[RuleID]:
        """
        Moves highly stable rules from RuleBase to ValueBase (Axioms).
        """
        promoted = []
        
        for rule_id, rule in theory.rules.items():
            # Criteria: High Confidence AND High Usage
            if rule.confidence >= self.promotion_threshold and rule.usage_count >= self.min_usage:
                # Promote!
                logger.info(f"CORTEX: Promoting {rule_id} to Axiom! (Conf={rule.confidence}, Usage={rule.usage_count})")
                
                # Just lock the rule in RuleBase (confidence = 1.0, immutable).
                rule.confidence = 1.0
                promoted.append(rule_id)
                
        return promoted

    def check_conflicts(self, theory: RuleBase) -> List[str]:
        """
        Detects logical conflicts (e.g., A and not A).
        """
        conflicts = []
        # O(N^2) check
        rules = list(theory.rules.values())
        for i in range(len(rules)):
            for j in range(i+1, len(rules)):
                r1 = rules[i]
                r2 = rules[j]
                
                if r1.head.predicate == r2.head.predicate and r1.head.args == r2.head.args:
                    if r1.head.negated != r2.head.negated:
                        # Opposite heads. Check bodies.
                        if self._bodies_equal(r1.body, r2.body):
                            conflicts.append(f"Conflict between {r1.id} and {r2.id}")
                            logger.warning(f"CORTEX: Logical Conflict detected! {r1.id} vs {r2.id}")
                            
        return conflicts

    def _bodies_equal(self, b1: List[Literal], b2: List[Literal]) -> bool:
        if len(b1) != len(b2): return False
        # Set comparison for order independence
        s1 = set(str(l) for l in b1)
        s2 = set(str(l) for l in b2)
        return s1 == s2
