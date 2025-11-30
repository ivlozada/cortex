from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from .types import FailureContext, Patch, PatchOperation
from .config import KernelConfig
from .selectors import DiscriminativeFeatureSelector
from .inference import InferenceEngine
from .rules import RuleBase

class RepairStrategy(ABC):
    """
    Abstract Base Class for Repair Strategies.
    Each strategy proposes patches to fix a specific type of logical failure.
    """
    def __init__(self, config: KernelConfig):
        self.config = config

    @abstractmethod
    def propose(self, ctx: FailureContext, features: Dict[str, Any]) -> List[Patch]:
        """
        Proposes a list of patches given the failure context and extracted features.
        """
        pass

    @property
    def name(self) -> str:
        return self.__class__.__name__

class NumericThresholdStrategy(RepairStrategy):
    """
    CORTEX-OMEGA v1.4: Numeric Threshold Learning.
    Uses DiscriminativeFeatureSelector.select_numeric_splits to propose
    numeric constraints like:

        attr(X, V_attr), V > 3.5

    as additional literals in the rule body.
    """

    def __init__(self, config: KernelConfig):
        super().__init__(config)
        self.selector = DiscriminativeFeatureSelector(min_score=0.02)

    def propose(self, ctx: FailureContext, features: Dict[str, Any]) -> List[Patch]:
        patches: List[Patch] = []

        # We only try this if we have memory; otherwise numeric stats are meaningless
        if not ctx.memory:
            return patches

        # Ask selector for best numeric splits per predicate
        splits = self.selector.select_numeric_splits(ctx)
        if not splits:
            return patches

        # We’ll just take the top few (3) to avoid explosion
        for pred, op, threshold, score in splits[:3]:
            # Canonical pattern:
            #   attr(X, V_attr), V_attr > threshold   OR   V_attr <= threshold
            # We implement this via ADD_LITERAL with "add_body" so PatchApplier
            # uses parse_literal on each string.
            var_name = f"V_{pred}"

            add_body = [
                f"{pred}(X, {var_name})",       # bind numeric value to var_name
                f"{var_name} {op} {threshold}", # numeric comparison literal
            ]

            patch = Patch(
                operation=PatchOperation.ADD_LITERAL,
                target_rule_id=str(ctx.rule.id),
                details={
                    "add_body": add_body,
                    "split_predicate": pred,
                    "operator": op,
                    "threshold": threshold,
                },
                confidence=float(score),
                explanation=(
                    f"Numeric split on {pred}: prefer cases where "
                    f"{var_name} {op} {threshold:.4f}"
                ),
                source_strategy=self.name
            )
            patches.append(patch)

        return patches

class TemporalConstraintStrategy(RepairStrategy):
    """
    CORTEX-OMEGA v1.4: Temporal Sequence Learning.
    If a rule fires on a negative example (False Positive), try to add a temporal constraint
    (T2 > T1) that holds for positives but fails for this negative.
    """
    def propose(self, ctx: FailureContext, features: Dict[str, Any]) -> List[Patch]:
        patches = []
        if ctx.error_type != "FALSE_POSITIVE":
            return []
            
        # 1. Find all variables in the rule body
        variables = set()
        for lit in ctx.rule.body:
            for arg in lit.args:
                if isinstance(arg, str) and arg and arg[0].isupper():
                    variables.add(arg)
                    
        if len(variables) < 2:
            return []
            
        # 2. Get bindings for the current negative firing
        engine = InferenceEngine(ctx.scene_facts, RuleBase())
        bindings_list = engine.evaluate_body(ctx.rule.body)
        
        if not bindings_list:
            return []
            
        # Check each grounding (usually just one for specific counter-example)
        for bindings in bindings_list:
            # Identify numeric values (timestamps)
            numeric_vars = []
            for var, val in bindings.items():
                try:
                    float(val)
                    numeric_vars.append((var, float(val)))
                except ValueError:
                    continue
            
            if len(numeric_vars) < 2:
                continue
                
            # Try all pairs
            for v1, val1 in numeric_vars:
                for v2, val2 in numeric_vars:
                    if v1 == v2: continue
                    
                    # We want a constraint that FAILS here (to kill the FP).
                    # So if we propose v1 > v2, it must be that val1 <= val2 currently.
                    if val1 <= val2:
                        # Potential constraint: v1 > v2
                        # Check if this holds for POSITIVES in memory.
                        
                        consistent_with_positives = True
                        if ctx.memory:
                            for s in ctx.memory:
                                if not s.ground_truth: continue
                                if s.target_predicate != ctx.target_predicate: continue
                                
                                # Find bindings for s
                                matcher_pos = InferenceEngine(s.facts, RuleBase())
                                pos_bindings_list = matcher_pos.evaluate_body(ctx.rule.body)
                                if not pos_bindings_list: continue 
                                
                                # Check if v1 > v2 holds for at least one grounding in this positive
                                satisfied_in_s = False

                                for pb in pos_bindings_list:
                                    try:
                                        pval1 = float(pb.get(v1, "0"))
                                        pval2 = float(pb.get(v2, "0"))
                                        if pval1 > pval2:
                                            satisfied_in_s = True
                                            break
                                    except:
                                        pass
                                
                                if not satisfied_in_s:
                                    consistent_with_positives = False
                                    break
                        
                        if consistent_with_positives:
                            # Found a valid temporal constraint!
                            patch = Patch(
                                operation=PatchOperation.ADD_LITERAL,
                                target_rule_id=str(ctx.rule.id),
                                details={
                                    "add_body": [f">({v1}, {v2})"]
                                },
                                confidence=self.config.hyperparams.temporal_confidence_threshold,
                                source_strategy=self.name
                            )
                            patches.append(patch)
                            
        return patches
