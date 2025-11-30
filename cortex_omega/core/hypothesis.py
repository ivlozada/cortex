"""
Hypothesis Generator v0.3 - El Sintetizador de Parches Lógicos (H_φ)
====================================================================
Este es el componente crítico que propone correcciones estructurales
a reglas que han fallado.

Arquitectura híbrida:
1. Analizador Heurístico: Extrae features del contexto de fallo
2. Generador de Candidatos: Propone patches basados en patrones
3. Ranker Neural (opcional): Ordena candidatos por probabilidad

Para la PoC v0.2, usamos heurísticas informadas + búsqueda acotada.
El componente neural se añade en v0.3.
"""

import logging
import math
import random
from collections import defaultdict

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional, Any, TYPE_CHECKING
from typing import Dict, List, Set, Tuple, Optional, Any, TYPE_CHECKING
from .rules import Rule, Literal, FactBase, Scene, RuleBase, RuleID, parse_literal
from .inference import InferenceEngine
from .types import Patch, PatchOperation, FailureContext
from .strategies import NumericThresholdStrategy, TemporalConstraintStrategy
import copy

logger = logging.getLogger(__name__)


from .selectors import DiscriminativeFeatureSelector
class FeatureExtractor:
    """
    Extrae features estructurales del contexto de fallo.
    Estas features guían la generación de hipótesis.
    """
    
    @staticmethod
    def extract(ctx: FailureContext) -> Dict[str, Any]:
        """Extrae features del contexto de fallo."""
        features = {
            "error_type": ctx.error_type,
            "rule_body_size": len(ctx.rule.body),
            "target_properties": {},
            "unused_predicates": set(),
            "related_entities": set(),
            "relational_facts": [],
            "discriminating_features": []
        }
        
        # CORTEX-OMEGA: Feature Selection
        selector = DiscriminativeFeatureSelector(min_score=0.1) # Higher threshold to avoid noise
        relevant_predicates = selector.select_features(ctx)
        
        # Propiedades del target
        for pred, args_set in ctx.scene_facts.facts.items():
            # CORTEX-OMEGA: Prevent Data Leakage
            if pred == ctx.target_predicate or pred == f"NOT_{ctx.target_predicate}":
                continue
                
            # Filter noise if we have enough data to know better
            if relevant_predicates and pred not in relevant_predicates:
                 # Keep core relations or if list is empty
                 if pred not in ["left_of", "behind", "above", "below"]:
                     continue
            for args in args_set:
                if ctx.target_entity in args:
                    if len(args) == 2:
                        # Binary predicate: p(X, Y) -> property p=Y
                        # Assuming target is first arg
                        if args[0] == ctx.target_entity:
                            # Excluir predicados relacionales de target_properties
                            if pred not in ["left_of", "behind", "above", "below"]:
                                features["target_properties"][pred] = args[1]
                            # Relación: el target está relacionado con otra entidad
                            if pred in ["behind", "left_of", "right_of", "above", "below"]:
                                features["relational_facts"].append({
                                    "predicate": pred,
                                    "related_entity": args[1]
                                })
                                features["related_entities"].add(args[1])
                        else:
                            # Relación inversa: p(Y, X) -> Y is related to X
                            features["related_entities"].add(args[0])
                    elif len(args) == 1:
                        # Unary predicate: p(X) -> property p=True
                        features["target_properties"][pred] = "true"
                    if len(args) == 2 and args[0] == ctx.target_entity:
                        # Relación: el target está relacionado con otra entidad
                        if pred in ["behind", "left_of", "right_of", "above", "below"]:
                            features["relational_facts"].append({
                                "predicate": pred,
                                "related_entity": args[1]
                            })
                            features["related_entities"].add(args[1])
        
        # Predicados usados en la regla vs disponibles
        rule_predicates = {lit.predicate for lit in ctx.rule.body}
        # Propiedades de entidades relacionadas
        features["related_properties"] = {}
        target_entity = ctx.target_entity
        # print(f"DEBUG: Extracting features for {target_entity}. Facts: {ctx.scene_facts.facts.keys()}")
        for rel_entity in features["related_entities"]:
            features["related_properties"][rel_entity] = {}
            for pred, args_set in ctx.scene_facts.facts.items():
                for args in args_set:
                    if rel_entity == args[0] and len(args) == 2:
                        features["related_properties"][rel_entity][pred] = args[1]
        
        return features


class HeuristicGenerator:
    """
    Generador de candidatos basado en heurísticas.
    Implementa patrones conocidos de reparación lógica.
    """
    
    # Prioridad de propiedades para excepciones (mayor = más relevante)
    # CORTEX-OMEGA: Standardized to Higher Value = Higher Priority
    # DEPRECATED: Use config.hyperparams.feature_weights instead
    # PROPERTY_PRIORITY = { ... }
    
    def __init__(self, config: 'KernelConfig', embedding_model=None):
        self.config = config
        self.embedding_model = embedding_model
        self.strategies = [
            TemporalConstraintStrategy(config), # CORTEX-OMEGA v1.4: Highest Priority (Temporal Logic)
            self._strategy_add_property_filter,
            self._strategy_add_negation,
            self._strategy_add_relational_constraint,
            self._strategy_specialize_with_exception,
            self._strategy_create_disjunction,
            self._strategy_create_concept,
            self._strategy_anti_unification,
            self._strategy_relational_anti_unification,
            self._strategy_contrastive_refinement, # NUEVO: Refinamiento contrastivo
            self._strategy_create_negative_rule,   # NUEVO: Explicit Negation
            NumericThresholdStrategy(config),  # CORTEX-OMEGA v1.4: Numeric Thresholds
        ]
        
        # CORTEX-OMEGA: Meta-Cognition
        self.success_history = {} # strategy_name -> success_count



    def _strategy_contrastive_refinement(self, ctx: FailureContext, features: Dict[str, Any]) -> List[Patch]:
        """
        Estrategia: Refinamiento Contrastivo.
        Para FALSE_POSITIVE: Buscar propiedades que tienen los positivos en memoria
        pero que le faltan al target.
        """
        patches = []
        if ctx.error_type != "FALSE_POSITIVE" or not ctx.memory:
            return []

        # 1. Recolectar ejemplos positivos relevantes
        positives = [s for s in ctx.memory if s.ground_truth and s.target_predicate == ctx.target_predicate]
        if not positives:
            return []

        # 2. Encontrar propiedades comunes en positivos
        # (Simplificado: propiedades que aparecen en al menos un positivo y NO en el target)
        
        candidate_literals = {} # literal_str -> count
        
        for pos_scene in positives:
            # Extraer propiedades del positivo
            ent = pos_scene.target_entity
            for pred, args_set in pos_scene.facts.facts.items():
                for args in args_set:
                    if ent in args:
                        # Construir literal candidato
                        if len(args) == 2 and args[0] == ent:
                            lit_key = (pred, args[1]) # (p, val)
                            candidate_literals[lit_key] = candidate_literals.get(lit_key, 0) + 1
                        elif len(args) == 1:
                            lit_key = (pred, "true")
                            candidate_literals[lit_key] = candidate_literals.get(lit_key, 0) + 1


        # 3. Filtrar propiedades que el target YA tiene
        target_props = features["target_properties"] # {pred: val}
        
        final_candidates = []
        for (pred, val), count in candidate_literals.items():
            # Si el target ya tiene esta propiedad con este valor, ignorar
            if pred in target_props and str(target_props[pred]) == str(val):
                continue
            
            # Calcular confianza basada en soporte
            confidence = 0.7 * (count / len(positives))
            
            final_candidates.append(((pred, val), confidence))

        # 4. Generar parches
        for (pred, val), conf in final_candidates:
            # Construir args para ADD_LITERAL
            if val == "true":
                args = ("X",)
            else:
                args = ("X", val)
                
            patch = Patch(
                operation=PatchOperation.ADD_LITERAL,
                target_rule_id=str(ctx.rule.id),
                details={
                    "predicate": pred,
                    "args": args
                },
                confidence=conf,
                explanation=f"Positivos tienen {pred}={val}, target no."
            )
            patches.append(patch)
            
        return patches

    def generate_candidates(self, context: FailureContext, features: Dict[str, Any]) -> List[Patch]:
        """
        Genera candidatos a parches usando heurísticas y CORTEX-OMEGA.
        """
        candidates = []
        
        # 1. CORTEX-OMEGA: Get Crystallized Priorities
        prioritized_strategies = []
        if hasattr(self, 'crystallizer'): # Check if parent has crystallizer (this is HeuristicGenerator, need to pass it down or access differently)
             # Wait, HeuristicGenerator is separate. I need to pass the crystallizer or the priorities to it.
             pass 
        
        # REFACTOR: Move Crystallizer logic to HypothesisGenerator's main loop or pass it here.
        # For now, let's assume HeuristicGenerator has access or we modify the caller.
        
        # Actually, let's modify HypothesisGenerator.generate_candidates (the caller) to sort the strategies?
        # No, HeuristicGenerator.generate_candidates does the work.
        
        # Let's inject crystallizer into HeuristicGenerator.
        strategies = self.strategies.copy()
        
        # If we have priorities passed in features (hacky but works) or if we attach crystallizer to HeuristicGenerator.
        if 'cortex_priorities' in features:
            priority_names = features['cortex_priorities']
            # Sort strategies: those in priority_names come first, in order.
            def get_strat_name(s):
                return s.__name__ if hasattr(s, '__name__') else s.name
            strategies.sort(key=lambda s: priority_names.index(get_strat_name(s)) if get_strat_name(s) in priority_names else 999)
            
        for strategy in strategies:
            if hasattr(strategy, 'propose'):
                # New Strategy Class
                patches = strategy.propose(context, features)
            else:
                # Legacy Method
                patches = strategy(context, features)
                
            for p in patches:
                if not p.source_strategy:
                    p.source_strategy = strategy.__name__ if hasattr(strategy, '__name__') else strategy.name
            candidates.extend(patches)
        
        # Ordenar por confianza
        candidates.sort(key=lambda p: p.confidence, reverse=True)
        
        return candidates
    


    def _strategy_add_property_filter(self, ctx: FailureContext, features: Dict[str, Any]) -> List[Patch]:
        """
        Estrategia: Añadir filtro de propiedad.
        Para FALSE_POSITIVE: añadir condición que el target NO cumple.
        Para FALSE_NEGATIVE: añadir condición que el target SÍ cumple.
        """
        patches = []
        
        if ctx.error_type == "FALSE_POSITIVE":
            # El target NO debería cumplir. Buscar propiedades que lo distinguen.
            # logger.debug(f"Checking target properties: {features['target_properties']}")
            for pred, value in features["target_properties"].items():
                # Si la propiedad es booleana "true", el literal es pred(X).
                # Si es valorada, es pred(X, val).
                
                # CORTEX-OMEGA v1.3: Apply Causal Priors
                priority = self.config.hyperparams.feature_weights.get(pred, 1.0)
                base_conf = 0.7 * priority
                
                # Penalize low confidence patches to avoid clutter
                if base_conf < 0.4:
                    continue

                if value == "true":
                    # Arity 1: negated literal NOT pred(X)
                    args = ("X",)
                    patch = Patch(
                        operation=PatchOperation.ADD_NEGATED_LITERAL,
                        target_rule_id=str(ctx.rule.id),
                        details={
                            "predicate": pred,
                            "args": args
                        },
                        confidence=base_conf,
                        explanation=f"Excluir si {pred}"
                    )
                else:
                    # Arity 2: negated literal NOT pred(X, value)
                    args = ("X", value)
                    patch = Patch(
                        operation=PatchOperation.ADD_NEGATED_LITERAL,
                        target_rule_id=str(ctx.rule.id),
                        details={
                            "predicate": pred,
                            "args": args
                        },
                        confidence=base_conf,
                        explanation=f"Excluir si {pred}={value}"
                    )
                patches.append(patch)

        elif ctx.error_type == "FALSE_NEGATIVE":
            # El target SÍ debería cumplir. Buscar propiedades comunes en los positivos.
            # O simplemente probar propiedades del target actual.
            for pred, value in features["target_properties"].items():
                
                # CORTEX-OMEGA v1.3: Apply Causal Priors
                priority = self.config.hyperparams.feature_weights.get(pred, 1.0)
                base_conf = 0.6 * priority
                
                if base_conf < 0.3:
                    continue

                if value == "true":
                    args = ("X",)
                else:
                    args = ("X", value)
                    
                patch = Patch(
                    operation=PatchOperation.ADD_LITERAL,
                    target_rule_id=str(ctx.rule.id),
                    details={
                        "predicate": pred,
                        "args": args
                    },
                    confidence=base_conf,
                    explanation=f"Considerar {pred}={value} como condición"
                )
                patches.append(patch)
    
        return patches
    
    def _strategy_add_negation(self, ctx: FailureContext, features: Dict[str, Any]) -> List[Patch]:
        """
        Estrategia: Añadir negación de relación.
        Especialmente útil para excepciones tipo "...UNLESS..."
        """
        patches = []
        
        if ctx.error_type == "FALSE_POSITIVE" and features["relational_facts"]:
            # El target tiene relaciones que quizás deberían bloquearlo
            for rel_fact in features["relational_facts"]:
                related = rel_fact["related_entity"]
                rel_pred = rel_fact["predicate"]
                
                # Propiedades de la entidad relacionada
                rel_props = features["related_properties"].get(related, {})
                
                # ORDENAR por prioridad (material > size > shape > color)
                sorted_props = sorted(
                    rel_props.items(),
                    key=lambda x: self.config.hyperparams.feature_weights.get(x[0], 99)
                )
                
                for i, (prop, value) in enumerate(sorted_props):
                    # Hipótesis: "no debería estar [rel] de algo [prop]=[value]"
                    # Dar mayor confianza a propiedades con mayor prioridad
                    base_confidence = 0.75
                    confidence = base_confidence - (i * 0.1)  # Decrece con prioridad
                    
                    patch = Patch(
                        operation=PatchOperation.ADD_EXCEPTION,
                        target_rule_id=str(ctx.rule.id),
                        details={
                            "relation": rel_pred,
                            "related_property": prop,
                            "related_value": value,
                            "pattern": f"NOT ({rel_pred}(X, Y) AND {prop}(Y, {value}))"
                        },
                        confidence=max(0.3, confidence),
                        explanation=f"Excepción: no aplica si está {rel_pred} de algo {prop}={value}"
                    )
                    patches.append(patch)
        
        return patches
    
    def _strategy_add_relational_constraint(self, ctx: FailureContext, features: Dict) -> List[Patch]:
        """
        Estrategia: La regla necesita una condición relacional.
        Útil para FALSE_NEGATIVE cuando hay una relación clave.
        """
        patches = []
        
        if ctx.error_type == "FALSE_NEGATIVE" and features["relational_facts"]:
            for rel_fact in features["relational_facts"]:
                related = rel_fact["related_entity"]
                rel_pred = rel_fact["predicate"]
                rel_props = features["related_properties"].get(related, {})
                
                for prop, value in rel_props.items():
                    patch = Patch(
                        operation=PatchOperation.ADD_LITERAL,
                        target_rule_id=str(ctx.rule.id),
                        details={
                            "add_body": [
                                f"{rel_pred}(X, Y)",
                                f"{prop}(Y, {value})"
                            ]
                        },
                        confidence=0.55,
                        explanation=f"Requiere estar {rel_pred} de algo {prop}={value}"
                    )
                    patches.append(patch)
        
        return patches
    
    def _strategy_specialize_with_exception(self, ctx: FailureContext, features: Dict) -> List[Patch]:
        """
        Estrategia: Crear cláusula de excepción explícita.
        Útil cuando el patrón general es correcto pero hay excepciones.
        """
        patches = []
        
        if ctx.error_type == "FALSE_POSITIVE":
            # Crear una regla auxiliar "abnormal(X) :- ..."
            # y modificar la principal a "glows(X) :- ..., not abnormal(X)"
            
            # Construir condiciones de anormalidad
            abnormal_conditions = []
            
            # 1. Propiedades directas del target
            existing_predicates = {lit.predicate for lit in ctx.rule.body}
            
            for prop, value in features["target_properties"].items():
                # CORTEX-OMEGA v1.3: Prevent Self-Contradiction
                # Do not use a property as an exception if it is already a condition for the rule!
                if prop in existing_predicates:
                    continue

                # Handle boolean normalization
                if value == "true":
                    abnormal_conditions.append({
                        "property": prop,
                        "value": None, # Unary
                        "is_unary": True
                    })
                else:
                    abnormal_conditions.append({
                        "property": prop,
                        "value": value,
                        "is_unary": False
                    })

            # 2. Propiedades relacionales
            for rel_fact in features["relational_facts"]:
                related = rel_fact["related_entity"]
                rel_pred = rel_fact["predicate"]
                rel_props = features["related_properties"].get(related, {})
                
                for prop, value in rel_props.items():
                    abnormal_conditions.append({
                        "relation": rel_pred,
                        "property": prop,
                        "value": value,
                        "is_unary": False
                    })
            
            if abnormal_conditions:
                patch = Patch(
                    operation=PatchOperation.ADD_EXCEPTION,
                    target_rule_id=str(ctx.rule.id),
                    details={
                        "create_abnormal_rule": True,
                        "abnormal_conditions": abnormal_conditions
                    },
                    confidence=0.7,
                    explanation="Crear regla de excepción explícita"
                )
                patches.append(patch)
        
        return patches
    
    def _strategy_create_disjunction(self, ctx: FailureContext, features: Dict) -> List[Patch]:
        """
        Estrategia: Crear rama alternativa (disyunción).
        Para FALSE_NEGATIVE: el target debería cumplir pero no cumple ninguna regla.
        En lugar de modificar la regla existente, creamos una NUEVA.
        """
        patches = []
        
        if ctx.error_type != "FALSE_NEGATIVE":
            return patches
        
        # Buscar propiedades distintivas del target que podrían formar una regla alternativa
        target_props = features["target_properties"]
        
        if not target_props:
            return patches
        
        # Crear candidatos de rama basados en propiedades del target
        # Priorizar por propiedades más distintivas (Higher Priority First)
        sorted_props = sorted(
            target_props.items(),
            key=lambda x: self.config.hyperparams.feature_weights.get(x[0], 0.0),
            reverse=True
        )
        
        for prop, value in sorted_props[:3]:  # Top 3 propiedades
            patch = Patch(
                operation=PatchOperation.CREATE_BRANCH,
                target_rule_id=ctx.rule.id,
                details={
                    "new_rule_body": [f"{prop}(X, {value})"],
                    "branch_property": prop,
                    "branch_value": value
                },
                confidence=0.65,
                explanation=f"Crear rama alternativa: {prop}={value} también implica glows"
            )
            patches.append(patch)
        
        # También considerar combinaciones de 2 propiedades
        if len(sorted_props) >= 2:
            prop1, val1 = sorted_props[0]
            prop2, val2 = sorted_props[1]
            patch = Patch(
                operation=PatchOperation.CREATE_BRANCH,
                target_rule_id=ctx.rule.id,
                details={
                    "new_rule_body": [f"{prop1}(X, {val1})", f"{prop2}(X, {val2})"],
                    "branch_property": f"{prop1}+{prop2}",
                    "branch_value": f"{val1}+{val2}"
                },
                confidence=0.55,
                explanation=f"Crear rama: {prop1}={val1} AND {prop2}={val2}"
            )
            patches.append(patch)
        
        return patches
    
    def _strategy_anti_unification(self, ctx: FailureContext, features: Dict) -> List[Patch]:
        """
        Estrategia: Anti-Unificación (Generalización).
        Busca ejemplos positivos en memoria que compartan estructura con el caso actual
        para proponer una regla general.
        """
        patches = []
        
        # Solo aplica para FALSE_NEGATIVE (queremos cubrir un caso nuevo)
        if ctx.error_type != "FALSE_NEGATIVE":
            return []
            
        # Si no hay memoria, no podemos generalizar
        if not ctx.memory:
            return []
            
        # Buscar ejemplos positivos en memoria para el mismo predicado target
        target_pred = ctx.target_predicate # Default to scene target
        if ctx.rule and ctx.rule.head:
            target_pred = ctx.rule.head.predicate
            
        # Filtrar memoria por mismo predicado target y ground truth True
        similar_scenes = []
        import random
        
        for s in ctx.memory:
            if s.ground_truth == True and s.target_predicate == target_pred:
                similar_scenes.append(s)
                
        if not similar_scenes:
            return []
            
        # Para cada escena similar, buscar intersección de hechos con la escena actual
        current_facts = ctx.scene_facts
        current_target = ctx.target_entity
        
        for past_scene in similar_scenes:
            past_facts = past_scene.facts
            past_target = past_scene.target_entity
            
            # Anti-unificación simple:
            # Buscar predicados que sean verdaderos para AMBOS targets
            common_predicates = []
            aux_rules_collected = [] # Reset for each scene
            variable_values = {} # var_name -> {"current": val, "past": val}
            
            # 1. Propiedades involving target
            for pred, args_set in current_facts.facts.items():
                for args in args_set:
                    if current_target in args:
                        # Construct generalized literal with X
                        gen_args = list(args)
                        # Replace target with "X"
                        for i, arg in enumerate(gen_args):
                            if arg == current_target:
                                gen_args[i] = "X"
                        gen_lit = Literal(pred, tuple(gen_args))
                        
                        # Check if past scene has equivalent fact
                        # We need to find if past_facts has pred(past_target, value)
                        # Construct expected past args
                        past_args_check = list(args)
                        for i, arg in enumerate(past_args_check):
                            if arg == current_target:
                                past_args_check[i] = past_target
                        
                        if past_facts.contains(Literal(pred, tuple(past_args_check))):
                            common_predicates.append(gen_lit)
                        else:
                            # CORTEX-OMEGA v1.4: Value Generalization
                            # If exact match fails, check if we can generalize a constant to a variable.
                            # Look for pred(past_target, V_past) in past_facts
                            # where structure matches but value differs.
                            
                            # Iterate over past facts to find structural match
                            for past_args in past_facts.facts.get(pred, []):
                                if len(past_args) != len(args): continue
                                
                                # Check if target position matches
                                match_structure = True
                                diff_indices = []
                                
                                for k, arg_k in enumerate(args):
                                    if arg_k == current_target:
                                        if past_args[k] != past_target:
                                            match_structure = False
                                            break
                                    else:
                                        if past_args[k] != arg_k:
                                            diff_indices.append(k)
                                            
                                if match_structure and len(diff_indices) == 1:
                                    # Found a match with exactly one difference (the value)
                                    # Generalize to variable
                                    gen_args_var = list(gen_args)
                                    idx = diff_indices[0]
                                    var_name = f"V_{pred}_{idx}" # Unique variable name
                                    gen_args_var[idx] = var_name
                                    
                                    # Store values for temporal analysis
                                    current_val = args[idx]
                                    past_val = past_args[idx]
                                    variable_values[var_name] = {"current": current_val, "past": past_val}
                                    
                                    new_lit = Literal(pred, tuple(gen_args_var))
                                    if new_lit not in common_predicates:
                                        common_predicates.append(new_lit)
                        
                        # Semantic match check (simplified for now, focusing on structure)
                        if self.embedding_model:
                             # ... (Semantic logic would go here, but let's stick to structural for Newton)
                             pass
            
            # 2. Global Facts (Constants): P(c) in both
            for pred, args_set in current_facts.facts.items():
                for args in args_set:
                    # Skip if it's the target entity (handled in step 1)
                    if args == (current_target,):
                        continue
                        
                    # Check if exact same fact exists in past
                    if past_facts.contains(Literal(pred, args)):
                        common_predicates.append(Literal(pred, args))

            if common_predicates:
                # CORTEX-OMEGA v1.4: Infer Temporal Constraints
                # Check ordering between numeric variables
                numeric_vars = []
                for var, vals in variable_values.items():
                    try:
                        c_val = float(vals["current"])
                        p_val = float(vals["past"])
                        numeric_vars.append((var, c_val, p_val))
                    except ValueError:
                        pass
                
                if len(numeric_vars) >= 2:
                    for i in range(len(numeric_vars)):
                        for j in range(len(numeric_vars)):
                            if i == j: continue
                            v1, c1, p1 = numeric_vars[i]
                            v2, c2, p2 = numeric_vars[j]
                            
                            # Check >
                            if c1 > c2 and p1 > p2:
                                common_predicates.append(Literal(">", (v1, v2)))
                            # Check < (redundant if we check > both ways, but explicit is fine)
                            # elif c1 < c2 and p1 < p2:
                            #     common_predicates.append(Literal("<", (v1, v2)))

                # Crear regla generalizada
                # R_gen: target(X) :- common_predicates(X)
                
                # Nombre único
                rule_id = f"R_gen_{len(common_predicates)}"
                new_rule = Rule(RuleID(rule_id), Literal(target_pred, ("X",)), common_predicates)
                
                patch = Patch(
                    operation=PatchOperation.CREATE_BRANCH,
                    target_rule_id=str(ctx.rule.id) if ctx.rule else "genesis",
                    details={
                        "rule": new_rule,
                        "aux_rules": aux_rules_collected,
                        "source_scene": past_scene.id
                    },
                    confidence=0.85 # Boost confidence for generalized rules
                )
                patches.append(patch)
        
        return patches

    
    def _strategy_relational_anti_unification(self, ctx: FailureContext, features: Dict) -> List[Patch]:
        """
        Estrategia: Anti-Unificación Relacional (Caminos).
        Busca caminos comunes entre entidades en ejemplos positivos.
        """
        patches = []
        if ctx.error_type != "FALSE_NEGATIVE":
            return []
            
        if not ctx.target_args or len(ctx.target_args) != 2:
            return []
            
        X0, Z0 = ctx.target_args
        logger.debug(f"DEBUG: Anti-Unification for {ctx.target_predicate}({X0}, {Z0})")
        
        # Helper to find paths of length 2: X -> Y -> Z
        def find_paths(facts: FactBase, start: str, end: str) -> Set[Tuple[str, str]]:
            paths = set()
            # Find all Y such that p(start, Y) and q(Y, end)
            # Iterate over all facts to find potential Ys
            # Optimization: First find all Ys connected to start
            
            # p(start, Y)
            first_hops = []
            for pred, args_set in facts.facts.items():
                # Sort for determinism
                for args in sorted(list(args_set)):
                    if len(args) == 2 and args[0] == start:
                        first_hops.append((pred, args[1])) # (p, Y)
            
            for p, Y in first_hops:
                # Check for q(Y, end)
                for pred, args_set in facts.facts.items():
                    # Check existence (no need to sort for existence check)
                    if (Y, end) in args_set:
                        paths.add((p, pred))
            return paths

        current_paths = find_paths(ctx.scene_facts, X0, Z0)
        logger.debug(f"DEBUG: Current paths: {current_paths}")
        if not current_paths:
            return []
            
        if not ctx.memory:
            return []
            
        for s in ctx.memory:
            if not s.ground_truth or s.target_predicate != ctx.target_predicate:
                continue
            if not s.target_args or len(s.target_args) != 2:
                continue
                
            X1, Z1 = s.target_args
            past_paths = find_paths(s.facts, X1, Z1)
            
            # Intersect
            common = current_paths.intersection(past_paths)
            logger.debug(f"DEBUG: Common paths with {s.id}: {common}")
            
            # Sort for determinism
            for p, q in sorted(list(common)):
                # Create rule: target(X, Z) :- p(X, Y), q(Y, Z)
                rule_id = f"R_rel_{p}_{q}"
                head = Literal(ctx.target_predicate, ("X", "Z"))
                body = [
                    Literal(p, ("X", "Y")),
                    Literal(q, ("Y", "Z"))
                ]
                new_rule = Rule(RuleID(rule_id), head, body)
                
                patch = Patch(
                    operation=PatchOperation.CREATE_BRANCH,
                    target_rule_id=str(ctx.rule.id) if ctx.rule else "genesis",
                    details={
                        "rule": new_rule,
                        "source_scene": s.id
                    },
                    confidence=0.9,
                    explanation=f"Relación transitiva encontrada: {p} -> {q}"
                )
                patches.append(patch)
                
        return patches

    def _strategy_create_concept(self, ctx: FailureContext, features: Dict) -> List[Patch]:
        """
        Estrategia: Crear concepto intermedio reutilizable.
        En lugar de añadir literales crudos a la regla, crea un predicado nuevo.
        Ej: near_blue(X) :- left_of(X, Y), color(Y, blue).
        """
        patches = []
        
        # Solo proponer conceptos si hay relaciones interesantes
        if not features["relational_facts"]:
            return patches
            
        # 1. Conceptos Relacionales (para FALSE_NEGATIVE o FALSE_POSITIVE)
        # Si la regla necesita una restricción relacional, mejor encapsularla.
        
        for rel_fact in features["relational_facts"]:
            related = rel_fact["related_entity"]
            rel_pred = rel_fact["predicate"]
            rel_props = features["related_properties"].get(related, {})
            
            for prop, value in rel_props.items():
                # Nombre sugerido para el concepto: near_blue, behind_metal, etc.
                # Mapeo simple de predicados espaciales a nombres legibles
                rel_name_map = {
                    "left_of": "near", "right_of": "near", 
                    "behind": "blocked_by", "above": "supported_by", "below": "supports"
                }
                prefix = rel_name_map.get(rel_pred, "related_to")
                concept_name = f"{prefix}_{value}"  # Ej: near_blue, blocked_by_metal
                
                # Crear parche
                patch = Patch(
                    operation=PatchOperation.CREATE_CONCEPT,
                    target_rule_id=str(ctx.rule.id),
                    details={
                        "concept_name": concept_name,
                        "variable": "X",
                        "body_literals": [
                            f"{rel_pred}(X, Y)",
                            f"{prop}(Y, {value})"
                        ],
                        "negated": (ctx.error_type == "FALSE_POSITIVE") # Negar si es para excluir
                    },
                    confidence=0.6, # Un poco más alto que ADD_LITERAL crudo por ser más limpio
                    explanation=f"Crear concepto {concept_name}(X) y usarlo"
                )
                patches.append(patch)
                
        return patches

    def _strategy_create_negative_rule(self, ctx: FailureContext, features: Dict) -> List[Patch]:
        """
        Estrategia: Crear Regla Negativa Explícita.
        Para NEGATIVE_GAP (True Negative pero sin explicación):
        Propone: Not Target(X) :- Body(X).
        """
        patches = []
        
        if ctx.error_type != "NEGATIVE_GAP":
            return patches
            
        # Buscar propiedades que expliquen por qué es NEGATIVO
        # Usamos las propiedades del target
        target_props = features["target_properties"]
        
        if not target_props:
            return patches
            
        # Priorizar propiedades distintivas
        sorted_props = sorted(
            target_props.items(),
            key=lambda x: self.config.hyperparams.feature_weights.get(x[0], 99)
        )
        
        for prop, value in sorted_props[:3]:
            # Crear regla: Not Target(X) :- Prop(X, Val)
            # Literal de cabeza negado
            head = Literal(ctx.target_predicate, ("X",), negated=True)
            body = [Literal(prop, ("X", value))]
            
            new_rule = Rule(RuleID(f"R_neg_{prop}_{value}"), head, body)
            
            patch = Patch(
                operation=PatchOperation.CREATE_BRANCH,
                target_rule_id="genesis", # New rule from scratch
                details={
                    "rule": new_rule,
                    "source_scene": "negative_gap"
                },
                confidence=0.85,
                explanation=f"Regla negativa explícita: {prop}={value} implica NO {ctx.target_predicate}"
            )
            patches.append(patch)
            
        return patches


class PatchApplier:
    """Aplica parches a reglas existentes."""
    
    @staticmethod
    def apply(rule: Rule, patch: Patch, rule_base: RuleBase = None) -> Tuple[Rule, List[Rule]]:
        """
        Aplica un parche y retorna la regla modificada + reglas auxiliares.
        
        Retorna:
            (regla_principal_modificada, [reglas_auxiliares_nuevas])
        """
        auxiliary_rules = []
        
        if patch.operation == PatchOperation.SEQUENCE:
            # Aplicar secuencia de parches
            current_rule = rule
            all_aux = []
            for sub_patch in patch.details["patches"]:
                current_rule, aux = PatchApplier.apply(current_rule, sub_patch, rule_base)
                all_aux.extend(aux)
            return current_rule, all_aux
            
        if patch.operation == PatchOperation.CREATE_BRANCH:
            # Crear nueva regla independiente
            details = patch.details
            
            # Si ya viene la regla pre-fabricada (ej. Anti-Unificación), usarla
            # Si ya viene la regla pre-fabricada (ej. Anti-Unificación), usarla
            if "rule" in details:
                aux = details.get("aux_rules", [])
                return details["rule"], aux
                
            branch_id = f"R_branch_{details.get('branch_property', 'new')}"
            
            new_body = []
            for lit_str in details.get("new_rule_body", []):
                new_body.append(parse_literal(lit_str))
            
            new_rule = Rule(
                id=RuleID(branch_id),
                head=copy.deepcopy(rule.head),
                body=new_body,
                confidence=patch.confidence
            )
            return new_rule, auxiliary_rules
        
        elif patch.operation == PatchOperation.CREATE_CONCEPT:
            # Crear regla auxiliar para el concepto y usarlo en la regla principal
            details = patch.details
            concept_name = details["concept_name"]
            
            # 1. Crear regla del concepto: concept(X) :- body...
            concept_body = []
            for lit_str in details["body_literals"]:
                concept_body.append(parse_literal(lit_str))
                
            concept_rule = Rule(
                id=RuleID(f"Concept_{concept_name}"),
                head=Literal(concept_name, (details["variable"],)),
                body=concept_body,
                confidence=1.0
            )
            auxiliary_rules.append(concept_rule)
            
            # 2. Modificar regla principal para usar el concepto
            new_rule = rule.clone(new_id=f"{rule.id}_v{rule.support_count + 1}")
            
            # CORTEX-OMEGA: Reset stats for new hypothesis
            new_rule.support_count = 0
            new_rule.failure_count = 0
            new_rule.fires_pos = 0
            new_rule.fires_neg = 0
            new_rule.confidence = 1.0 # Optimistic start
            
            new_literal = Literal(
                concept_name, 
                (details["variable"],), 
                negated=details.get("negated", False)
            )
            new_rule.body.append(new_literal)
            
            return new_rule, auxiliary_rules
        
        # Para otras operaciones, clonar y modificar
        new_rule = rule.clone(new_id=f"{rule.id}_v{rule.support_count + 1}")
        
        # CORTEX-OMEGA: Reset stats for new hypothesis
        new_rule.support_count = 0
        new_rule.failure_count = 0
        new_rule.fires_pos = 0
        new_rule.fires_neg = 0
        new_rule.confidence = 1.0 # Optimistic start
        
        if patch.operation == PatchOperation.ADD_LITERAL:
            details = patch.details
            if "predicate" in details:
                new_literal = Literal(
                    details["predicate"],
                    details["args"],
                    details.get("negated", False)
                )
                new_rule.body.append(new_literal)
            elif "add_body" in details:
                for lit_str in details["add_body"]:
                    new_rule.body.append(parse_literal(lit_str))
        
        elif patch.operation == PatchOperation.ADD_NEGATED_LITERAL:
            details = patch.details
            new_literal = Literal(
                details["predicate"],
                details["args"],
                negated=True
            )
            new_rule.body.append(new_literal)
        
        elif patch.operation == PatchOperation.ADD_EXCEPTION:
            details = patch.details
            
            if "pattern" in details:
                # NUEVO: Crear predicado auxiliar CON su definición semántica
                blocker_pred = f"blocked_by_{details['related_property']}"
                
                # 1. Añadir negación del blocker a la regla principal
                new_literal = Literal(blocker_pred, ("X",), negated=True)
                new_rule.body.append(new_literal)
                
                # 2. CREAR LA REGLA QUE DEFINE EL BLOCKER (¡la clave!)
                # blocked_by_material(X) :- behind(X, Y), material(Y, metal)
                blocker_rule = Rule(
                    id=RuleID(f"{blocker_pred}_def"),
                    head=Literal(blocker_pred, ("X",)),
                    body=[
                        Literal(details["relation"], ("X", "Y")),
                        Literal(details["related_property"], ("Y", details["related_value"]))
                    ],
                    confidence=1.0
                )
                auxiliary_rules.append(blocker_rule)
            
            elif "create_abnormal_rule" in details:
                # Crear regla abnormal con todas las condiciones
                # CORTEX-OMEGA v1.3: Dynamic Naming
                abnormal_pred = f"abnormal_for_{rule.head.predicate}"
                new_literal = Literal(abnormal_pred, ("X",), negated=True)
                new_rule.body.append(new_literal)
                
                # Crear regla que define abnormal basada en condiciones
                # Crear regla que define abnormal basada en condiciones
                for i, cond in enumerate(details.get("abnormal_conditions", [])):
                    body = []
                    if "relation" in cond:
                        body.append(Literal(cond["relation"], ("X", "Y")))
                        if cond.get("is_unary"):
                             body.append(Literal(cond["property"], ("Y",)))
                        else:
                             body.append(Literal(cond["property"], ("Y", cond["value"])))
                    else:
                        # Direct property
                        if cond.get("is_unary"):
                            body.append(Literal(cond["property"], ("X",)))
                        else:
                            body.append(Literal(cond["property"], ("X", cond["value"])))
                            
                    abnormal_rule = Rule(
                        id=RuleID(f"{abnormal_pred}_cond_{i}"),
                        head=Literal(abnormal_pred, ("X",)),
                        body=body,
                        confidence=1.0
                    )
                    auxiliary_rules.append(abnormal_rule)
        
        elif patch.operation == PatchOperation.REMOVE_LITERAL:
            idx = patch.details.get("index", -1)
            if 0 <= idx < len(new_rule.body):
                new_rule.body.pop(idx)
        
        return new_rule, auxiliary_rules


class PatternCrystallizer:
    """
    CORTEX-OMEGA Pillar 1: Pattern Crystallizer.
    Learns from repair history to prioritize successful strategies.
    """
    def __init__(self):
        # Map: ContextHash -> {StrategyName: SuccessCount}
        self.patterns = {}
        # Map: ContextHash -> TotalAttempts
        self.context_stats = {}

    def _get_context_hash(self, features: Dict[str, Any]) -> str:
        """Creates a hashable signature of the error context."""
        # We focus on the error type and the shape of the problem
        keys = sorted(features.keys())
        signature = []
        for k in keys:
            val = features[k]
            if isinstance(val, (int, float, str, bool)):
                signature.append(f"{k}:{val}")
            elif isinstance(val, list):
                signature.append(f"{k}:len({len(val)})")
        return "|".join(signature)

    def record_success(self, features: Dict[str, Any], strategy_name: str):
        """Records a successful repair strategy for a given context."""
        ctx_hash = self._get_context_hash(features)
        
        if ctx_hash not in self.patterns:
            self.patterns[ctx_hash] = {}
            self.context_stats[ctx_hash] = 0
            
        if strategy_name not in self.patterns[ctx_hash]:
            self.patterns[ctx_hash][strategy_name] = 0
            
        self.patterns[ctx_hash][strategy_name] += 1
        self.context_stats[ctx_hash] += 1
        logger.info(f"CORTEX: Crystallized pattern! Context[{ctx_hash[:20]}...] -> {strategy_name} (+1)")

    def get_priorities(self, features: Dict[str, Any]) -> List[str]:
        """Returns strategies sorted by historical success rate."""
        ctx_hash = self._get_context_hash(features)
        if ctx_hash not in self.patterns:
            return []
            
        stats = self.patterns[ctx_hash]
        # Sort by count descending
        sorted_strategies = sorted(stats.items(), key=lambda x: x[1], reverse=True)
        return [s[0] for s in sorted_strategies]


class AnalogicalMemory:
    """
    CORTEX-OMEGA Pillar 2: Analogical Bootstrapping.
    Stores successful repairs and retrieves them based on feature similarity.
    """
    def __init__(self):
        # List of (features, patch) tuples
        self.memory: List[Tuple[Dict, Patch]] = []

    def add(self, features: Dict[str, Any], patch: Patch):
        """Stores a successful repair case."""
        # Store a deep copy to avoid mutation issues
        self.memory.append((copy.deepcopy(features), copy.deepcopy(patch)))
        logger.info(f"CORTEX: Stored analogical case. Total memory: {len(self.memory)}")

    def retrieve(self, current_features: Dict[str, Any], threshold: float = 0.7) -> List[Patch]:
        """
        Retrieves patches from similar past contexts.
        Uses a weighted Jaccard similarity on features.
        """
        candidates = []
        
        for past_features, past_patch in self.memory:
            similarity = self._compute_similarity(current_features, past_features)
            
            if similarity >= threshold:
                logger.info(f"CORTEX: Found analogy! Similarity: {similarity:.2f}")
                # Adapt the patch to the current context
                adapted_patch = self._adapt_patch(past_patch, past_features, current_features)
                if adapted_patch:
                    # Boost confidence based on similarity
                    adapted_patch.confidence = max(0.8, similarity)
                    adapted_patch.explanation = f"[Analogy] Based on similar case (sim={similarity:.2f}): {past_patch.explanation}"
                    candidates.append(adapted_patch)
                    
        return candidates

    def _compute_similarity(self, f1: Dict, f2: Dict) -> float:
        """Computes weighted similarity between two feature sets."""
        # Simplified Jaccard-like similarity
        # We focus on keys that match and have similar values
        
        score = 0.0
        total_weight = 0.0
        
        # Weights for different feature types
        weights = {
            "error_type": 2.0,
            "target_predicate": 1.0,
            "rule_body_size": 0.5,
            "discriminating_features": 1.5
        }
        
        all_keys = set(f1.keys()) | set(f2.keys())
        
        for k in all_keys:
            w = weights.get(k, 1.0)
            total_weight += w
            
            v1 = f1.get(k)
            v2 = f2.get(k)
            
            if v1 == v2:
                score += w
            elif isinstance(v1, (list, set)) and isinstance(v2, (list, set)):
                # Overlap for lists/sets
                s1 = set(v1)
                s2 = set(v2)
                if s1 and s2:
                    overlap = len(s1 & s2) / len(s1 | s2)
                    score += w * overlap
            # Else mismatch contributes 0
            
        return score / total_weight if total_weight > 0 else 0.0

    def _adapt_patch(self, patch: Patch, source_features: Dict, target_features: Dict) -> Optional[Patch]:
        """
        Adapts a patch from source context to target context.
        For v1, we focus on simple structural transfer (reusing the operation).
        """
        # Deep copy the patch to avoid modifying memory
        new_patch = copy.deepcopy(patch)
        
        # If the patch adds a literal, we might need to map variables or values.
        # For now, we assume the variable names (X, Y) are consistent across problems (canonical form).
        # But values (e.g., 'red', 'sphere') might need to change.
        
        # Heuristic: If the patch involves a value that was present in the source features
        # but is NOT in the target features, we might need to swap it.
        # However, finding the *correct* swap is hard without a mapping.
        
        # Strategy for v1:
        # 1. If it's a structural change (e.g., CREATE_BRANCH), just return it (it triggers a strategy).
        # 2. If it's adding a specific filter (ADD_LITERAL), check if the literal makes sense.
        
        # Actually, the most robust way for v1 is to return the *Strategy Name* 
        # and let the strategy re-run on the new context.
        # But 'Patch' objects are the output of strategies.
        
        # Let's try to adapt the specific content if possible.
        if patch.operation == PatchOperation.ADD_LITERAL:
            # Check if the literal uses constants
            lit = patch.details.get("literal") # This might be a Literal object or dict
            # If it's a dict (serialized)
            pass 
            
        # For v1, we will trust that if the context is similar enough, the patch might apply directly
        # OR we rely on the fact that we are prioritizing the *strategy* via Crystallizer.
        # Wait, Analogy is supposed to be better than Crystallizer.
        # It should transfer *specifics*.
        
        # Example: "Add filter color(X, red)" -> "Add filter color(X, blue)"
        # If source error was on "red object" and target is "blue object".
        
        return new_patch


class ComplexityEstimator:
    """
    CORTEX-OMEGA Pillar 3: Structural Gradient.
    Estimates the structural cost of a patch before application.
    Gradient = Confidence / (1 + lambda * Cost)
    """
    def __init__(self):
        # Base costs for operations
        self.costs = {
            PatchOperation.ADD_LITERAL: 1.0,
            PatchOperation.REMOVE_LITERAL: 1.0,
            PatchOperation.ADD_NEGATED_LITERAL: 1.5, # Slightly more complex
            PatchOperation.CREATE_BRANCH: 5.0,       # High cost: increases search space
            PatchOperation.ADD_EXCEPTION: 3.0,       # Moderate cost
            PatchOperation.CREATE_CONCEPT: 4.0,      # High cost: adds vocabulary
            PatchOperation.SEQUENCE: 0.0             # Sum of parts
        }

    def estimate(self, patch: Patch) -> float:
        """Returns the estimated complexity cost of a patch."""
        if patch.operation == PatchOperation.SEQUENCE:
            total = 0.0
            if "patches" in patch.details:
                for p in patch.details["patches"]:
                    if isinstance(p, Patch):
                        total += self.estimate(p)
            return total
            
        cost = self.costs.get(patch.operation, 2.0)
        
        # Refinements based on details
        if patch.operation == PatchOperation.ADD_LITERAL:
            # Adding a literal with many arguments is more complex?
            # For now, constant cost.
            pass
            
        return cost


class HypothesisGenerator:
    """
    El componente H_φ completo.
    Orquesta extracción de features, generación de candidatos y aplicación.
    """
    
    def __init__(self, config: 'KernelConfig', neural_ranker=None, disable_motifs=False, disable_concept_invention=False, search_strategy="beam", embedding_model=None, lambda_complexity=0.3):
        self.config = config
        self.feature_extractor = FeatureExtractor()
        self.heuristic_generator = HeuristicGenerator(config=config, embedding_model=embedding_model)
        
        # ABLATION: Disable Concept Invention
        if disable_concept_invention:
            # Remove _strategy_create_concept from strategies
            self.heuristic_generator.strategies = [
                s for s in self.heuristic_generator.strategies 
                if s.__name__ != "_strategy_create_concept"
            ]
            
        self.patch_applier = PatchApplier()
        self.neural_ranker = neural_ranker
        
        # Biblioteca de Motivos (v0.4)
        try:
            from .motif_library import MotifLibrary
            self.motif_library = MotifLibrary()
        except ImportError:
            logger.warning("motif_library not found. Disabling motifs.")
            self.motif_library = None
            disable_motifs = True
            
        # CORTEX-OMEGA: Pattern Crystallizer
        self.crystallizer = PatternCrystallizer()
        
        # CORTEX-OMEGA: Analogical Memory
        self.analogical_memory = AnalogicalMemory()
        
        # CORTEX-OMEGA: Structural Gradient
        self.complexity_estimator = ComplexityEstimator()
        self.lambda_complexity = lambda_complexity
        
        # ABLATION: Disable Motifs
        self.disable_motifs = disable_motifs
        

        
        # ABLATION: Search Strategy
        self.search_strategy = search_strategy
        
        # Historial para meta-aprendizaje
        self.repair_history: List[Dict] = []
    
    def diagnose(self, ctx: FailureContext) -> Dict:
        """Analiza un fallo y extrae diagnóstico."""
        features = self.feature_extractor.extract(ctx)
        
        diagnosis = {
            "error_type": ctx.error_type,
            "features": features,
            "likely_causes": []
        }
        
        # Inferir causas probables
        if ctx.error_type == "FALSE_POSITIVE":
            if features["relational_facts"]:
                diagnosis["likely_causes"].append(
                    "Missing exception for relational constraint"
                )
            diagnosis["likely_causes"].append(
                "Rule too general - needs additional conditions"
            )
        else:
            if not ctx.rule.body:
                diagnosis["likely_causes"].append("Rule body is empty")
            diagnosis["likely_causes"].append(
                "Rule too specific - needs generalization or branch"
            )
        
        return diagnosis
    
    def generate(self, ctx: FailureContext, top_k: int = 5, beam_width: int = 1) -> List[Tuple[Patch, Rule, List[Rule]]]:
        """
        Genera candidatos de parche para corregir el error en el contexto dado.
        """
        import math
        import sys
        

        # 0. Extract Features
        features = self.feature_extractor.extract(ctx)
        
        # Si beam_width > 1, usar búsqueda en haz
        # ABLATION: Force greedy if strategy is 'greedy'
        if beam_width > 1 and self.search_strategy != "greedy":
            candidates = self.beam_search(ctx, beam_width=beam_width)
        else:
            # Modo legacy / greedy
            
            # CORTEX-OMEGA: Get priorities
            priorities = self.crystallizer.get_priorities(features)
            if priorities:
                features['cortex_priorities'] = priorities
                logger.info(f"CORTEX: Applying crystallized wisdom. Prioritizing: {priorities}")
                
            # CORTEX-OMEGA: Analogical Retrieval
            analogical_candidates = self.analogical_memory.retrieve(features)
            if analogical_candidates:
                logger.info(f"CORTEX: Retrieved {len(analogical_candidates)} analogical candidates.")
                
            raw_candidates = self.heuristic_generator.generate_candidates(ctx, features)
            
            # Merge candidates (Analogy first)
            all_raw_candidates = analogical_candidates + raw_candidates
            
            candidates = []
            seen_rules = set()
            
            for patch in all_raw_candidates:
                # CORTEX-OMEGA: Structural Gradient
                # Gradient = Confidence / (1 + lambda * Cost)
                cost = self.complexity_estimator.estimate(patch)
                gradient_score = patch.confidence / (1.0 + self.lambda_complexity * cost)
                
                # Store score for debugging/sorting
                patch.details['gradient_score'] = gradient_score
                patch.details['cost'] = cost
                
                try:
                    new_rule, aux_rules = self.patch_applier.apply(ctx.rule, patch)
                    rule_str = str(new_rule)
                    if rule_str in seen_rules:
                        continue
                    seen_rules.add(rule_str)
                    
                    candidates.append((patch, new_rule, aux_rules))
                except Exception:
                    continue

            # Sort by Gradient Score instead of raw confidence
            candidates.sort(key=lambda x: x[0].details.get('gradient_score', 0.0), reverse=True)

        # Si hay ranker entrenado, re-pondera las confianzas
        if self.neural_ranker is not None and getattr(self.neural_ranker, "trained", False):
            features = self.feature_extractor.extract(ctx) # Asegurar features
            scores = self.neural_ranker.score_candidates(ctx, candidates, features)
            
            # Re-asignar confianza
            for i, (patch, _, _) in enumerate(candidates):
                # Combinación: 0.4 heurística + 0.6 neural (con sigmoide)
                # scores[i] es logit
                p_neural = 1.0 / (1.0 + math.exp(-scores[i]))
                patch.confidence = 0.4 * patch.confidence + 0.6 * p_neural
            
            # Re-ordenar
            candidates.sort(key=lambda x: x[0].confidence, reverse=True)
            
        return candidates[:top_k]

    def beam_search(self, ctx: FailureContext, beam_width: int = 3, max_depth: int = 2) -> List[Tuple[Patch, Rule, List[Rule]]]:
        """
        Búsqueda en haz para encontrar secuencias de parches.
        """
        # Estado: (rule, aux_rules, accumulated_patches, score)
        # Fallback to greedy for now to unblock
        features = self.feature_extractor.extract(ctx)
        
        # CORTEX-OMEGA: Get priorities
        priorities = self.crystallizer.get_priorities(features)
        if priorities:
            features['cortex_priorities'] = priorities
            
        raw_candidates = self.heuristic_generator.generate_candidates(ctx, features)
        candidates = []
        for patch in raw_candidates:
            try:
                new_rule, aux_rules = self.patch_applier.apply(ctx.rule, patch)
                candidates.append((patch, new_rule, aux_rules))
            except Exception:
                continue
        return candidates[:beam_width]
        
    def feedback(self, winning_theory, context: FailureContext):
        """
        CORTEX-OMEGA: Receive feedback on which hypothesis won.
        """
        # We need to know WHICH strategy produced the winning theory.
        # This requires tracking the source strategy of each candidate.
        # For now, we'll assume the 'patch' object has metadata or we can infer it.
        pass

    def record_feedback(self, ctx: FailureContext, winning_patch: Patch):
        """
        Records the success of a patch.
        """
        features = self.feature_extractor.extract(ctx)
        # We assume the patch has a 'strategy_name' attribute or similar.
        # If not, we need to add it during generation.
        if hasattr(winning_patch, 'source_strategy'):
             self.crystallizer.record_success(features, winning_patch.source_strategy)
        else:
             logger.warning("Winning patch has no source_strategy. Cannot crystallize.")
             
        # CORTEX-OMEGA: Store in Analogical Memory
        self.analogical_memory.add(features, winning_patch)
        

    
    def record_outcome(self, patch: Patch, success: bool, ctx: FailureContext = None, features: Dict = None):
        """Registra el resultado de un parche para meta-aprendizaje."""
        
        # Serializar detalles (especialmente para SEQUENCE que tiene parches anidados)
        details = copy.deepcopy(patch.details)
        if patch.operation == PatchOperation.SEQUENCE:
            if "patches" in details:
                serialized_patches = []
                for p in details["patches"]:
                    if isinstance(p, Patch):
                        serialized_patches.append({
                            "operation": p.operation.value,
                            "details": p.details,
                            "confidence": p.confidence,
                            "explanation": p.explanation
                        })
                    else:
                        serialized_patches.append(p)
                details["patches"] = serialized_patches

        record = {
            "operation": patch.operation.value,
            "details": details,
            "success": success
        }
        
        if ctx:
            record.update({
                "error_type": ctx.error_type,
                "rule_id": str(ctx.rule.id),
                "rule_body": [str(l) for l in ctx.rule.body],
                "target_predicate": ctx.rule.head.predicate if ctx.rule.head else "unknown",
            })
        
        if features:
            record["features"] = features
            
        self.repair_history.append(record)
    
    def get_statistics(self) -> Dict:
        """Retorna estadísticas de reparación."""
        if not self.repair_history:
            return {"total": 0}
        
        by_operation = {}
        for record in self.repair_history:
            op = record["operation"]
            if op not in by_operation:
                by_operation[op] = {"attempts": 0, "successes": 0}
            by_operation[op]["attempts"] += 1
            if record["success"]:
                by_operation[op]["successes"] += 1
        
        for op in by_operation:
            attempts = by_operation[op]["attempts"]
            successes = by_operation[op]["successes"]
            by_operation[op]["success_rate"] = successes / attempts if attempts > 0 else 0
        
        return {
            "total": len(self.repair_history),
            "by_operation": by_operation
        }

    def save_history(self, filepath: str, append: bool = False):
        """Guarda el historial de reparaciones en JSONL."""
        import json
        mode = 'a' if append else 'w'
        with open(filepath, mode) as f:
            for record in self.repair_history:
                # Copia para no modificar el original
                rec = dict(record)
                
                # Convertir sets a listas en features
                if "features" in rec:
                    feats = dict(rec["features"])
                    for key in ["unused_predicates", "related_entities"]:
                        if key in feats and isinstance(feats[key], set):
                            feats[key] = list(feats[key])
                    rec["features"] = feats
                
                f.write(json.dumps(rec) + '\n')


# === Demo ===

if __name__ == "__main__":
    print("=== Hypothesis Generator Demo ===\n")
    
    # Crear contexto de fallo simulado
    facts = FactBase()
    facts.add("color", ("o1", "red"))
    facts.add("shape", ("o1", "sphere"))
    facts.add("material", ("o2", "metal"))
    facts.add("behind", ("o1", "o2"))
    facts.add("size", ("o1", "large"))
    
    naive_rule = Rule(
        id=RuleID("R1"),
        head=Literal("glows", ("X",)),
        body=[Literal("color", ("X", "red"))]
    )
    
    ctx = FailureContext(
        rule=naive_rule,
        error_type="FALSE_POSITIVE",
        target_entity="o1",
        scene_facts=facts,
        prediction=True,
        ground_truth=False
    )
    
    print(f"Regla actual: {naive_rule}")
    print(f"Target: o1")
    print(f"Error: FALSE_POSITIVE (predijo True, debía ser False)")
    print(f"\nHechos de la escena:")
    print(facts)
    
    # Generar hipótesis
    from .config import KernelConfig
    config = KernelConfig()
    h_phi = HypothesisGenerator(config=config)
    
    diagnosis = h_phi.diagnose(ctx)
    print(f"\n=== Diagnóstico ===")
    print(f"Causas probables: {diagnosis['likely_causes']}")
    
    candidates = h_phi.generate(ctx, top_k=3)
    print(f"\n=== Candidatos de Parche ({len(candidates)}) ===")
    for i, (patch, new_rule, aux_rules) in enumerate(candidates):
        print(f"\n{i+1}. {patch}")
        print(f"   Regla resultante: {new_rule}")
        if aux_rules:
            print(f"   Reglas auxiliares: {len(aux_rules)}")
