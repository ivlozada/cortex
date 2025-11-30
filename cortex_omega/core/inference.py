"""
Cortex Core: Inference Engine
=============================
Forward chaining inference engine with support for negation.
"""

from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass
from .rules import Literal, Rule, FactBase, RuleBase, Scene, Term
from .config import KernelConfig
import logging

logger = logging.getLogger(__name__)
from collections import defaultdict

@dataclass
class DependencyIndex:
    by_predicate: Dict[str, List[Rule]]

    @classmethod
    def from_theory(cls, theory: RuleBase) -> "DependencyIndex":
        index = defaultdict(list)
        for rule in theory.rules.values():
            # print(f"DEBUG: Indexing rule {rule.id}: {rule}")
            for lit in rule.body:
                index[lit.predicate].append(rule)
        return cls(by_predicate=dict(index))
    
    def get_triggered_rules(self, predicate: str) -> List[Rule]:
        return self.by_predicate.get(predicate, [])

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
        self.dependency_index = DependencyIndex.from_theory(rule_base)

    
    def unify(self, literal: Literal, fact_args: Tuple[Any, ...]) -> Optional[Dict[str, Any]]:
        """Intenta unificar un literal con un hecho. Retorna bindings o None."""
        if len(literal.args) != len(fact_args):
            return None
        
        bindings = {}
        for lit_arg, fact_arg in zip(literal.args, fact_args):
            if not self._unify_term(lit_arg, fact_arg, bindings):
                return None
        return bindings

    def _get_term_depth(self, term: Term) -> int:
        if not term.args: return 1
        max_depth = 0
        for arg in term.args:
            if isinstance(arg, Term):
                d = self._get_term_depth(arg)
                if d > max_depth: max_depth = d
        return 1 + max_depth
            
    def _unify_term(self, term1: Any, term2: Any, bindings: Dict[str, Any]) -> bool:
        """Recursive unification of two terms."""
        # 1. Handle Variables in term1
        if isinstance(term1, str) and term1 and term1[0].isupper():
            return self._bind(term1, term2, bindings)
        if isinstance(term1, Term) and term1.is_variable():
            return self._bind(term1.name, term2, bindings)
            
        # 2. Handle Variables in term2 (if we support bidirectional unification later)
        # For now, fact_args are usually ground, but let's be safe
        
        # 3. Handle Terms
        if isinstance(term1, Term) and isinstance(term2, Term):
            if term1.name != term2.name:
                return False
            if len(term1.args) != len(term2.args):
                return False
            for a1, a2 in zip(term1.args, term2.args):
                if not self._unify_term(a1, a2, bindings):
                    return False
            return True
            
        # 4. Handle Strings (Legacy Constants)
        if isinstance(term1, str) and isinstance(term2, str):
            return term1 == term2
            
        # 5. Mixed types (Term vs String) -> Fail unless one is a variable (handled above)
        # Actually, Term("a") should match "a"? No, let's keep them distinct for now.
        return term1 == term2

    def _bind(self, var_name: str, value: Any, bindings: Dict[str, Any]) -> bool:
        if var_name in bindings:
            # Check consistency
            return bindings[var_name] == value
            
        # Occurs Check: Prevent X = s(X)
        if isinstance(value, Term) and self._occurs_check(var_name, value, bindings):
            return False
            
        bindings[var_name] = value
        return True
        
    def _occurs_check(self, var_name: str, term: Term, bindings: Dict[str, Any]) -> bool:
        if term.is_variable():
            if term.name == var_name:
                return True
            # Follow binding if exists
            if term.name in bindings:
                val = bindings[term.name]
                if isinstance(val, Term):
                    return self._occurs_check(var_name, val, bindings)
            return False
            
        for arg in term.args:
            if isinstance(arg, Term) and self._occurs_check(var_name, arg, bindings):
                return True
        return False
    
    def evaluate_body(self, body: List[Literal], bindings: Dict[str, str] = None) -> List[Dict[str, str]]:
        """
        Evalúa el cuerpo de una regla (conjunción de literales).
        Retorna una lista de bindings que satisfacen el cuerpo.
        """
        if bindings is None:
            bindings = {}
        # print(f"DEBUG: evaluate_body {body} bindings={bindings}")
        # Base case: empty body -> one success (current bindings)
        if not body:
            return [bindings]
            
        first = body[0]
        rest = body[1:]
        
        results = []
        
        # 1. Find bindings for the first literal
        candidates = self._get_bindings_for_literal(first, bindings)
        
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
        # print(f"DEBUG: ground_literal {ground_literal}")
        
        # 2. Query facts
        # DEBUG:
        # print(f"[ENGINE] ground_literal: {ground_literal}")
        # print(f"[ENGINE] facts for {ground_literal.predicate}: {self.facts.facts.get(ground_literal.predicate, set())}")
        
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
            
            # CORTEX-OMEGA: Sanitize args for query (Variables -> None)
            query_args = []
            for arg in ground_literal.args:
                if isinstance(arg, str) and arg and arg[0].isupper():
                    query_args.append(None)
                elif isinstance(arg, Term) and arg.is_variable():
                    query_args.append(None)
                else:
                    query_args.append(arg)
            
            for fact_args in self.facts.query(ground_literal.predicate, tuple(query_args)):
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
        # logger.debug(f"forward_chain started with max_iterations={max_iterations}")

        # Stratification
        early_rules = [r for r in self.rules.rules.values() if not any(l.negated for l in r.body)]
        all_rules = list(self.rules.rules.values())
        
        derived = set()

        def run_phase(rules_subset, phase_name):
            # Initial set of rules to check: ALL rules in the subset
            # Because we need to match against initial facts
            rules_to_check = set(r.id for r in rules_subset)
            
            iteration = 0
            while iteration < max_iterations:
                iteration += 1
                iteration_new_facts = []
                
                # 1. Evaluate rules
                # Only evaluate rules that are in rules_to_check AND in rules_subset
                # (rules_subset defines the phase scope)
                
                # Convert IDs back to Rule objects
                current_rules = []
                for rid in rules_to_check:
                    if rid in self.rules.rules:
                        rule = self.rules.rules[rid]
                        # Check if rule is in the current phase subset
                        # Optimization: Pre-compute IDs for phase subset
                        # For now, just check if it's in rules_subset list (slow O(N))
                        # Better: pass set of IDs to run_phase
                        current_rules.append(rule)
                
                # Filter by phase (naive but correct)
                # Actually, rules_subset is a list. Let's make it a set of IDs for O(1) check
                subset_ids = set(r.id for r in rules_subset)
                current_rules = [r for r in current_rules if r.id in subset_ids]
                
                # Clear rules_to_check for next iteration (unless we find new facts)
                next_rules_to_check = set()
                
                for rule in current_rules:
                    if debug: logger.debug(f"Evaluating rule {rule.id}: {rule}")
                    rule_new_count = 0
                    body_results = self.evaluate_body(rule.body)
                    
                    for bindings in body_results:
                        head_ground = rule.head.ground(bindings)
                        
                        # DEBUG: Check for monster terms
                        for arg in head_ground.args:
                            d = self._get_term_depth(arg)
                            if isinstance(arg, Term) and d > 200:
                                logger.error(f"Monster term generated by rule {rule.id}: {arg._repr_safe(0)} (Depth: {d})")
                                raise ValueError(f"Term too deep! Rule: {rule.id}")
                        
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
                    
                    # Update usage stats - REMOVED (Handled in engine.py)
                    # if rule_new_count > 0:
                    #     rule.support_count += 1
                    #     s = rule.support_count
                    #     f = rule.failure_count
                    #     rule.confidence = (s + 1.0) / (s + f + 2.0)

                if not iteration_new_facts:
                    if debug:
                        logger.debug(
                            f"forward_chain ({phase_name}): punto fijo alcanzado tras {iteration} iteraciones."
                        )
                    break # No new facts derived in this iteration, reach fixed point
                
                # 2. Add new facts to the fact base and provenance
                for head, rule_id, bindings in iteration_new_facts:
                    fact_added = False
                    predicate_key = ""
                    
                    if head.negated:
                        key = f"NOT_{head.predicate}"
                        if head.args not in self.facts.facts[key]:
                            self.facts.add(key, head.args)
                            fact_added = True
                            predicate_key = key
                    else:
                        if head.args not in self.facts.facts[head.predicate]:
                            self.facts.add(head.predicate, head.args)
                            fact_added = True
                            predicate_key = head.predicate
                    
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
                        
                        # Trigger dependent rules
                        # If we added P(a), trigger rules with P in body
                        # Note: NOT_P triggers rules with NOT_P in body
                        # Our index keys are predicates.
                        # If predicate_key is "NOT_P", we look for "NOT_P" in index?
                        # Literal.predicate for negated literal is just "P" but negated=True.
                        # DependencyIndex indexes by lit.predicate.
                        # Wait, DependencyIndex implementation:
                        # index[lit.predicate].append(rule)
                        # If lit is NOT P, lit.predicate is P.
                        # So if we add NOT_P, we should trigger rules that have NOT P.
                        # But if we index by P, we trigger rules with P AND rules with NOT P.
                        # That's fine (over-approximation).
                        
                        # However, self.facts stores "NOT_P" as a key.
                        # predicate_key is "NOT_P" or "P".
                        # If "NOT_P", the predicate name is "P" (conceptually).
                        # But our Literal structure keeps "P" and negated=True.
                        
                        trigger_pred = head.predicate
                        triggered = self.dependency_index.get_triggered_rules(trigger_pred)
                        for r in triggered:
                            next_rules_to_check.add(r.id)
                    
                    derived.add(head)
                    self.trace.append({
                        "type": "derivation",
                        "rule_id": rule_id,
                        "derived": head,
                        "bindings": bindings
                    })
                
                rules_to_check = next_rules_to_check

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

def infer(theory: RuleBase, scene: Scene, config: Optional[KernelConfig] = None) -> Tuple[bool, List[Dict[str, Any]]]:
    """
    Realiza inferencia sobre una escena usando la teoría dada.
    Retorna (predicción, traza).
    """
    engine = InferenceEngine(scene.facts, theory)
    max_iter = config.inference_max_iterations if config else 1000
    engine.forward_chain(max_iterations=max_iter)
    
    # Verificar si el target fue derivado
    if scene.target_args:
        target = Literal(scene.target_predicate, scene.target_args)
    else:
        target = Literal(scene.target_predicate, (scene.target_entity,))
        
    # CORTEX-OMEGA v1.3: Conflict Resolution
    pos_proof = engine.get_proof(target)
    
    if scene.target_args:
        neg_target = Literal(f"NOT_{scene.target_predicate}", scene.target_args)
    else:
        neg_target = Literal(f"NOT_{scene.target_predicate}", (scene.target_entity,))
    neg_proof = engine.get_proof(neg_target)
    
    # logger.debug(f"DEBUG: infer({scene.target_predicate}) -> {prediction}")
    prediction = False
    trace = getattr(engine, "trace", [])
    
    if pos_proof and not neg_proof:
        prediction = True
        trace = []
        for step in pos_proof.steps:
            trace.append({
                "type": "derivation",
                "rule_id": step.rule_id,
                "derived": step.derived_fact,
                "bindings": step.bindings
            })
    elif not pos_proof and neg_proof:
        prediction = False
        trace = []
        for step in neg_proof.steps:
            trace.append({
                "type": "derivation",
                "rule_id": step.rule_id,
                "derived": step.derived_fact,
                "bindings": step.bindings
            })
    elif pos_proof and neg_proof:
        if pos_proof.confidence > neg_proof.confidence:
            prediction = True
        logger.debug(f"Inference Trace for {scene.target_entity}:")
        for step in trace:
            logger.debug(f"  - {step}")
        import time
        for step in trace:
            if step["type"] == "derivation":
                rid = step["rule_id"]
                if rid in theory.rules:
                    theory.rules[rid].usage_count += 1
                    theory.rules[rid].last_used = time.time()
    
    return prediction, trace

def update_rule_stats(theory: RuleBase, trace: List[Dict[str, Any]], is_correct: bool, config: KernelConfig):
    """
    CORTEX-OMEGA v1.4: First-Class Rule Statistics.
    Updates fires_pos, fires_neg, support, failure, and confidence for ALL firing rules.
    """
    updated = set()
    for step in trace:
        if step["type"] == "derivation":
            rid = step["rule_id"]
            if rid in theory.rules and rid not in updated:
                rule = theory.rules[rid]
                
                if is_correct:
                    rule.fires_pos += 1
                    rule.support_count += 1
                else:
                    rule.fires_neg += 1
                    
                    # CORTEX-OMEGA v1.3: Mode-based Punishment
                    mode = config.mode
                    if mode == "strict":
                        rule.failure_count = 1_000_000
                        logger.info(f"💀 STRICT MODE: Rule {rid} killed due to counter-example.")
                    else:
                        rule.failure_count += 1
                
                # Bayesian Update
                s = rule.support_count
                f = rule.failure_count
                rule.confidence = (s + 1.0) / (s + f + 2.0)
                    
                updated.add(rid)



