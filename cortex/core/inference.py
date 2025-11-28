"""
Cortex Core: Inference Engine
=============================
Forward chaining inference engine with support for negation.
"""

from typing import Dict, List, Set, Tuple, Optional, Any
from .rules import Literal, Rule, FactBase, RuleBase

class InferenceEngine:
    """Motor de inferencia: forward chaining con soporte para negación."""
    
    def __init__(self, fact_base: FactBase, rule_base: RuleBase):
        self.facts = fact_base
        self.rules = rule_base
        self.trace: List[Dict] = []  # Traza de inferencia para credit assignment
    
    def unify(self, literal: Literal, fact_args: Tuple[str, ...]) -> Optional[Dict[str, str]]:
        """Intenta unificar un literal con un hecho. Retorna bindings o None."""
        if len(literal.args) != len(fact_args):
            return None
        
        bindings = {}
        for lit_arg, fact_arg in zip(literal.args, fact_args):
            if lit_arg[0].isupper():  # Es variable
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
        """Evalúa el cuerpo de una regla, retornando todas las sustituciones válidas."""
        if not body:
            return [initial_bindings or {}]
        
        results = [initial_bindings or {}]
        
        for literal in body:
            new_results = []
            
            for bindings in results:
                ground_literal = literal.ground(bindings)
                
                if ground_literal.negated:
                    # Negación por fallo: éxito si NO está en los hechos
                    if ground_literal.is_ground():
                        positive = Literal(ground_literal.predicate, ground_literal.args, False)
                        if not self.facts.contains(positive):
                            new_results.append(bindings)
                    else:
                        # Variable en negación: falla si existe algún binding
                        has_match = False
                        for fact_args in self.facts.query(literal.predicate):
                            if self.unify(Literal(literal.predicate, literal.args, False), fact_args):
                                has_match = True
                                break
                        if not has_match:
                            new_results.append(bindings)
                else:
                    # Literal positivo: buscar matches
                    for fact_args in self.facts.query(literal.predicate):
                        new_bindings = self.unify(ground_literal, fact_args)
                        if new_bindings is not None:
                            merged = {**bindings, **new_bindings}
                            new_results.append(merged)
            
            results = new_results
            if not results:
                break
        
        return results
    
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
        Inferencia hacia adelante con:
          - límite de iteraciones (max_iterations)
          - logging resumido por iteración y por regla
          - parada por punto fijo (cuando no se derivan hechos nuevos)
        """
        if logger is None:
            import logging
            logger = logging.getLogger(__name__)

        derived = set()

        iteration = 0
        while True:
            iteration += 1
            if iteration > max_iterations:
                if debug:
                    logger.warning(
                        "forward_chain: se alcanzó max_iterations=%d, deteniendo.",
                        max_iterations,
                    )
                break

            iteration_new_facts = [] # Lista de (head, rule_id, bindings)
            
            if debug and iteration <= log_first_n_iterations:
                logger.debug("=== forward_chain iteración %d ===", iteration)

            # Para cada regla, contamos cuántos hechos nuevos genera en ESTA iteración
            for rule in self.rules.rules.values():
                rule_new_count = 0

                # Lógica de aplicación de la regla
                body_results = self.evaluate_body(rule.body)
                
                for bindings in body_results:
                    head_ground = rule.head.ground(bindings)
                    
                    if not self.facts.contains(head_ground) and head_ground not in derived:
                        is_duplicate = False
                        for h, _, _ in iteration_new_facts:
                            if h == head_ground:
                                is_duplicate = True
                                break
                        
                        if not is_duplicate:
                            iteration_new_facts.append((head_ground, rule.id, bindings))
                            rule_new_count += 1

                if debug and iteration <= log_first_n_iterations and rule_new_count > 0:
                    logger.debug(
                        "Regla %s produjo %d hechos nuevos en iteración %d.",
                        rule.id,
                        rule_new_count,
                        iteration,
                    )

            if not iteration_new_facts:
                if debug:
                    logger.debug(
                        "forward_chain: punto fijo alcanzado tras %d iteraciones. Total hechos derivados: %d.",
                        iteration,
                        len(derived),
                    )
                break

            for head, rule_id, bindings in iteration_new_facts:
                self.facts.add(head.predicate, head.args)
                derived.add(head)
                self.trace.append({
                    "type": "derivation",
                    "rule_id": rule_id,
                    "derived": head,
                    "bindings": bindings
                })

            if debug and iteration <= log_first_n_iterations:
                logger.debug(
                    "Iteración %d: %d hechos nuevos (total acumulado = %d).",
                    iteration,
                    len(iteration_new_facts),
                    len(derived),
                )

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
