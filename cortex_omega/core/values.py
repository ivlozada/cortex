"""
Cortex Core: Value System
=========================
Axiomatic constraints that define "Goodness" for the agent.
"""

from dataclasses import dataclass
from typing import List, Dict, Set, Optional, TYPE_CHECKING
from .rules import Literal, FactBase

if TYPE_CHECKING:
    from .rules import RuleBase, Scene

@dataclass
class Axiom:
    """
    A normative constraint.
    Format: "condition -> forbidden"
    If 'condition' is true, then 'forbidden' must NOT be true.
    """
    name: str
    condition: List[Literal] # e.g. [fragile(X), heavy(X)]
    forbidden: Literal # e.g. broken(X)
    description: str

class ValueBase:
    """
    The repository of sacred values.
    """
    def __init__(self):
        self.axioms: List[Axiom] = []
        
    def add_axiom(self, axiom: Axiom):
        self.axioms.append(axiom)
        
    def check_sin(self, facts: FactBase, prediction: Literal) -> Optional[Axiom]:
        """
        Checks if a prediction violates any axiom given the current facts.
        Returns the violated Axiom if any, else None.
        """
        for axiom in self.axioms:
            # Check if prediction matches forbidden pattern
            if prediction.predicate != axiom.forbidden.predicate:
                continue
            if len(prediction.args) != len(axiom.forbidden.args):
                continue
                
            # Bind variables
            # Ex: forbidden=glows(X), prediction=glows(o1) -> {X: o1}
            bindings = {}
            match = True
            for var, val in zip(axiom.forbidden.args, prediction.args):
                if var.startswith('X') or var.startswith('Y') or var.startswith('Z'): # Simple variable check
                    if var in bindings and bindings[var] != val:
                        match = False; break
                    bindings[var] = val
                else:
                    if var != val: # Constant mismatch
                        match = False; break
            
            if not match:
                continue
                
            # Check ALL conditions with bindings
            all_conditions_met = True
            
            # Handle single Literal case for backward compatibility if needed, 
            # but we are changing the type definition so let's assume list.
            conditions = axiom.condition if isinstance(axiom.condition, list) else [axiom.condition]
            
            for condition in conditions:
                cond_args = []
                for arg in condition.args:
                    if arg in bindings:
                        cond_args.append(bindings[arg])
                    else:
                        cond_args.append(arg)
                
                ground_condition = Literal(condition.predicate, tuple(cond_args), condition.negated)
                
                if not facts.contains(ground_condition):
                    all_conditions_met = False
                    # print(f"DEBUG: Condition {ground_condition} not found.")
                    break
            
            if all_conditions_met:
                # print(f"DEBUG: Sin detected! {axiom.name}")
                return axiom
            
        return None

    def evaluate_theory(self, rule_base: 'RuleBase', test_scenes: List['Scene']) -> float:
        """
        Evaluates the 'moral score' of a theory.
        1.0 = Saint
        0.0 = Sinner
        """
        # Placeholder for complex moral evaluation
        return 1.0
