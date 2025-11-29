"""
Cortex Core: Inference Engine
=============================
Forward chaining inference engine with support for negation.
"""

from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass
from .rules import Literal, Rule, FactBase, RuleBase

@dataclass
class ProofStep:
    """A single step in a logical proof."""
    derived_fact: Literal
    rule_id: str
    rule_str: str
    bindings: Dict[str, str]
    antecedents: List[Literal]

@dataclass
class Proof:
    """A full logical proof."""
    steps: List[ProofStep]
    confidence: float

    def __repr__(self):
        if not self.steps:
            return "Proof(Empty)"
        # Show the chain of derivations
        return " -> ".join([str(s.derived_fact) for s in self.steps])


class InferenceEngine:
    """Motor de inferencia: forward chaining con soporte para negación."""
    
    def __init__(self, fact_base: FactBase, rule_base: RuleBase):
        self.facts = fact_base
        self.rules = rule_base
        self.trace: List[Dict] = []  # Traza de inferencia para credit assignment
        # Map: derived_literal -> (rule_id, bindings, list_of_antecedent_literals)
        self.provenance: Dict[Literal, Tuple[str, Dict[str, str], List[Literal]]] = {}

    
    def unify(self, literal: Literal, fact_args: Tuple[str, ...]) -> Optional[Dict[str, str]]:
        """Intenta unificar un literal con un hecho. Retorna bindings o None."""
        if len(literal.args) != len(fact_args):
            return None
        
        bindings = {}
        for lit_arg, fact_arg in zip(literal.args, fact_args):
            if isinstance(lit_arg, str) and lit_arg and lit_arg[0].isupper():  # Es variable
                if lit_arg in bindings:
                    if bindings[lit_arg] != fact_arg:
                        return None
                else:
                    bindings[lit_arg] = fact_arg
            else:  # Es constante
                if lit_arg != fact_arg:
                    return None
        return bindings
    
    def evaluate_body(self, body: List[Literal], initial_bindings: Dict[str, str] = None) -> List[Dict[str, str]]:
        """
        Evalúa el cuerpo de una regla (conjunción de literales).
        Retorna una lista de bindings que satisfacen el cuerpo.
        """
        if initial_bindings is None:
            initial_bindings = {}
            
        # Base case: empty body -> one success (current bindings)
        if not body:
            return [initial_bindings]
            
        first = body[0]
        rest = body[1:]
        
        results = []
        
        # 1. Find bindings for the first literal
        candidates = self._get_bindings_for_literal(first, initial_bindings)
        
        for binding in candidates:
            # 2. Recursive step
            # Apply binding to the rest
            rest_ground = [lit.ground(binding) for lit in rest]
            
            rest_results = self.evaluate_body(rest_ground, binding) # Pass current binding
            
            for rest_binding in rest_results:
                # Merge bindings (should be consistent since we passed binding down)
                combined = {**binding, **rest_binding}
                results.append(combined)
                
        return results

    def _get_bindings_for_literal(self, literal: Literal, current_bindings: Dict[str, str]) -> List[Dict[str, str]]:
        """Helper to find bindings for a single literal, considering current_bindings."""
        
        # Apply current_bindings to the literal first
        ground_literal = literal.ground(current_bindings)
        
        # CORTEX-OMEGA: Built-in operator support
        if ground_literal.predicate in {">", "<", ">=", "<=", "=", "!="}:
            if ground_literal.is_ground():
                try:
                    # Attempt numeric conversion
                    def to_num(x):
                        try: return float(x)
                        except ValueError: return x
                    
                    val1 = to_num(ground_literal.args[0])
                    val2 = to_num(ground_literal.args[1])
                    
                    op = ground_literal.predicate
                    res = False
                    if op == ">": res = val1 > val2
                    elif op == "<": res = val1 < val2
                    elif op == ">=": res = val1 >= val2
                    elif op == "<=": res = val1 <= val2
                    elif op == "=": res = val1 == val2
                    elif op == "!=": res = val1 != val2
                    
                    if res:
                        return [current_bindings] # Success, return current bindings
                except Exception:
                    pass # Fail silently on type errors
            return [] # No match or not ground
        
        if ground_literal.negated:
            # Negación por fallo: éxito si NO está en los hechos
            if ground_literal.is_ground():
                positive = Literal(ground_literal.predicate, ground_literal.args, False)
                if not self.facts.contains(positive):
                    return [current_bindings] # Success, return current bindings
            else:
                # Variable en negación: falla si existe algún binding
                has_match = False
                for fact_args in self.facts.query(literal.predicate): # Use original literal for query
                    if self.unify(Literal(literal.predicate, literal.args, False), fact_args):
                        has_match = True
                        break
                if not has_match:
                    return [current_bindings] # Success, return current bindings
            return [] # Match found or not ground
        else:
            # Literal positivo: buscar matches
            found_bindings = []
            for fact_args in self.facts.query(ground_literal.predicate):
                new_bindings = self.unify(ground_literal, fact_args)
                if new_bindings is not None:
                    merged = {**current_bindings, **new_bindings}
                    found_bindings.append(merged)
            return found_bindings
    
    def query(self, literal: Literal, explain: bool = False) -> Tuple[bool, Optional[Dict]]:
        """
        Consulta si un literal puede derivarse.
        Retorna (resultado, explicación).
        """
        self.trace = []
        
        # Caso base: es un hecho directo
        if self.facts.contains(literal):
            if explain:
                return True, {"type": "fact", "literal": literal}
            return True, None
        
        # Buscar reglas que deriven este predicado
        for rule in self.rules.get_rules_for_predicate(literal.predicate):
            # Unificar la cabeza con el query
            head_bindings = self.unify(rule.head, literal.args)
            if head_bindings is None:
                continue
            
            # Evaluar el cuerpo con estos bindings
            body_results = self.evaluate_body(rule.body, head_bindings)
            
            if body_results:
                trace_entry = {
                    "type": "rule",
                    "rule_id": rule.id,
                    "rule": str(rule),
                    "bindings": body_results[0],
                    "literal": literal
                }
                self.trace.append(trace_entry)
                
                if explain:
                    return True, trace_entry
                return True, None
        
        return False, None
    
    def forward_chain(
        self,
        max_iterations: int = 1000,
        debug: bool = False,
        logger=None,
        log_first_n_iterations: int = 5,
    ) -> Set[Literal]:
        """
        Ejecuta forward chaining estratificado (Stratified Forward Chaining).
        Phase 1: Reglas sin negación (Early).
        Phase 2: Todas las reglas (Late).
        """
        if logger is None:
            import logging
            logger = logging.getLogger(__name__)

        # Stratification
        early_rules = [r for r in self.rules.rules.values() if not any(l.negated for l in r.body)]
        all_rules = list(self.rules.rules.values())
        
        derived = set()

        def run_phase(rules_subset, phase_name):
            iteration = 0
            while iteration < max_iterations:
                iteration += 1
                iteration_new_facts = []
                
                # 1. Evaluate rules
                for rule in rules_subset:
                    if debug: logger.debug(f"Evaluating rule {rule.id}: {rule}")
                    rule_new_count = 0
                    body_results = self.evaluate_body(rule.body)
                    
                    for bindings in body_results:
                        head_ground = rule.head.ground(bindings)
                        
                        # If head is already in facts, skip
                        if self.facts.contains(head_ground):
                            continue
                        
                        # Check duplicates in current batch
                        is_duplicate = False
                        for h, _, _ in iteration_new_facts:
                            if h == head_ground:
                                is_duplicate = True
                                break
                        
                        if not is_duplicate:
                            iteration_new_facts.append((head_ground, rule.id, bindings))
                            rule_new_count += 1
                    
                    # Update usage stats
                    if rule_new_count > 0:
                        rule.support_count += 1
                        s = rule.support_count
                        f = rule.failure_count
                        rule.confidence = (s + 1.0) / (s + f + 2.0)

                if not iteration_new_facts:
                    if debug:
                        logger.debug(
                            f"forward_chain ({phase_name}): punto fijo alcanzado tras {iteration} iteraciones."
                        )
                    break # No new facts derived in this iteration, reach fixed point
                
                # 2. Add new facts to the fact base and provenance
                for head, rule_id, bindings in iteration_new_facts:
                    fact_added = False
                    if head.negated:
                        key = f"NOT_{head.predicate}"
                        if head.args not in self.facts.facts[key]:
                            self.facts.add(key, head.args)
                            fact_added = True
                    else:
                        if head.args not in self.facts.facts[head.predicate]:
                            self.facts.add(head.predicate, head.args)
                            fact_added = True
                    
                    if fact_added:
                        # Normalize head for provenance
                        if head.negated:
                            provenance_head = Literal(f"NOT_{head.predicate}", head.args, negated=False)
                        else:
                            provenance_head = head
                            
                        # Reconstruct antecedents
                        antecedents = []
                        for lit in self.rules.rules[rule_id].body:
                            ground_lit = lit.ground(bindings)
                            antecedents.append(ground_lit)
                            
                        self.provenance[provenance_head] = (rule_id, bindings, antecedents)
                    
                    derived.add(head)
                    self.trace.append({
                        "type": "derivation",
                        "rule_id": rule_id,
                        "derived": head,
                        "bindings": bindings
                    })

        # Run Phases
        if debug: logger.debug("=== Phase 1: Early Rules (No Negation) ===")
        run_phase(early_rules, "Early")
        
        if debug: logger.debug("=== Phase 2: All Rules ===")
        run_phase(all_rules, "Late")
        
        return derived
    
    def predict(self, entity: str, target_predicate: str) -> Tuple[bool, List[Dict]]:
        """
        Predice si un predicado es verdadero para una entidad.
        Retorna (predicción, traza de inferencia).
        """
        self.trace = []
        target = Literal(target_predicate, (entity,))
        
        # Derivar todo lo posible
        self.forward_chain()
        
        # Verificar si el target fue derivado
        result = self.facts.contains(target)
        
        return result, self.trace
    def get_proof(self, target: Literal) -> Optional[Proof]:
        """
        Reconstructs the proof for a target literal using the provenance graph.
        """
        if not self.facts.contains(target):
            return None
            
        steps = []
        visited = set()
        
        def backtrack(current_lit: Literal):
            if current_lit in visited:
                return
            visited.add(current_lit)
            
            # If it's a base fact (no provenance), it's a leaf
            if current_lit not in self.provenance:
                return
                
            rule_id, bindings, antecedents = self.provenance[current_lit]
            
            # Recursively prove antecedents first
            for ant in antecedents:
                backtrack(ant)
                
            # Then add this step
            rule = self.rules.rules.get(rule_id)
            step = ProofStep(
                derived_fact=current_lit,
                rule_id=rule_id,
                rule_str=str(rule) if rule else "Unknown",
                bindings=bindings,
                antecedents=antecedents
            )
            steps.append(step)

        backtrack(target)
        
        if not steps:
            # It might be a base fact
            return Proof(steps=[], confidence=1.0)
            
        # Confidence is min of rule confidences in the chain (weakest link)
        min_conf = 1.0
        for step in steps:
            rule = self.rules.rules.get(step.rule_id)
            if rule:
                min_conf = min(min_conf, rule.confidence)
                
        return Proof(steps=steps, confidence=min_conf)
