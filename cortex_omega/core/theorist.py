from dataclasses import dataclass
from typing import List, Tuple, Any, Optional, Dict
import copy
import logging

from .rules import RuleBase, Scene, Rule, Literal, RuleID
from .config import KernelConfig
from .hypothesis import HypothesisGenerator
from .types import Patch, FailureContext

logger = logging.getLogger(__name__)

@dataclass
class Theorist:
    config: KernelConfig
    hypothesis_generator: HypothesisGenerator

    def propose(
        self, 
        theory: RuleBase, 
        scene: Scene, 
        memory: List[Scene], 
        trace, 
        error_type: str
    ) -> Tuple[List[Tuple[RuleBase, Any]], Optional[FailureContext]]:
        return generate_structural_candidates(
            theory=theory,
            scene=scene,
            memory=memory,
            trace=trace,
            error_type=error_type,
            config=self.config,
            feature_priors=self.config.feature_priors
        )

def generate_structural_candidates(
    theory: RuleBase,
    scene: Scene,
    memory: List[Scene],
    trace,
    error_type: str,
    config: KernelConfig,
    feature_priors: Optional[Dict[str, float]] = None,
) -> Tuple[List[Tuple[RuleBase, Any]], Optional[FailureContext]]:
    """
    Toma la teoría actual + la escena conflictiva y devuelve una lista de teorías candidatas
    ya parcheadas (RuleBase clonadas y modificadas).
    El kernel no sabe de 'Patch', solo ve teorías completas.
    """
    patch_generator = config.patch_generator
    if patch_generator is None:
        patch_generator = HypothesisGenerator(config=config)

    culprit_rule = identify_culprit_rule(
        theory, 
        error_type, 
        trace, 
        target_predicate=scene.target_predicate if scene.target_predicate else "glows"
    )
    
    # Handle Cold Start (No rules yet) or Failure to Identify Culprit
    if culprit_rule is None:
        if error_type in ["FALSE_NEGATIVE", "NEGATIVE_GAP"]:
            # Genesis Mode: Create a dummy rule to serve as the "culprit" for CREATE_BRANCH
            # This allows the generator to propose a new rule from scratch.
            # Use the scene's target predicate, not hardcoded 'glows'
            target_pred = scene.target_predicate if scene.target_predicate else "glows"
            
            # Determine variables based on arity
            if scene.target_args and len(scene.target_args) == 2:
                vars = ("X", "Z")
            else:
                vars = ("X",)
                
            culprit_rule = Rule(RuleID("dummy_genesis"), Literal(target_pred, vars), [])
        else:
            return [], None

    # Construir contexto de fallo (muy parecido a tu FailureContext actual)
    ctx = FailureContext(
        rule=culprit_rule,
        error_type=error_type,
        target_entity=scene.target_entity,
        target_predicate=scene.target_predicate,
        target_args=scene.target_args,
        scene_facts=scene.facts,
        memory=memory,
        prediction=(error_type == "FALSE_POSITIVE"),
        ground_truth=not (error_type == "FALSE_POSITIVE"),
        inference_trace=trace,
        feature_priors=feature_priors or {}
    )

    # Pass lambda_complexity to generator if supported
    if hasattr(patch_generator, 'lambda_complexity'):
        patch_generator.lambda_complexity = config.lambda_complexity

    raw_candidates = patch_generator.generate(ctx, top_k=20)
    candidate_theories: List[Tuple[RuleBase, Any]] = []

    for patch, new_rule, aux_rules in raw_candidates:
        # CORTEX-OMEGA: Safety Check
        # Reject rules with unbound head variables (unsafe)
        if not new_rule.is_safe:
            continue
            
        if any(not aux.is_safe for aux in aux_rules):
            continue

        # Clonar teoría completa
        T_candidate = copy.deepcopy(theory)

        # Aplicar la cirugía estructural de forma *local* a esta copia
        if patch.operation.value == "CREATE_BRANCH":
            # Añadir regla nueva sin eliminar la original
            T_candidate.add(new_rule)
        else:
            # Reemplazar la regla culpable
            # Si es genesis (dummy), no hay nada que reemplazar, solo añadir
            if culprit_rule.id == "dummy_genesis":
                T_candidate.add(new_rule)
            elif culprit_rule.id in T_candidate.rules:
                T_candidate.replace(culprit_rule.id, new_rule)
        
        candidate_theories.append((T_candidate, patch))
        for aux in aux_rules:
            if aux.id in T_candidate.rules:
                T_candidate.replace(aux.id, aux)
            else:
                T_candidate.add(aux)

        candidate_theories.append((T_candidate, patch))

    return candidate_theories, ctx


def identify_culprit_rule(
    theory: RuleBase,
    error_type: str,
    trace,
    target_predicate: str,
) -> Optional[Rule]:
    """
    Versión funcional de tu _identify_culprit, pero sin dependencias de 'self'.
    """
    if error_type == "FALSE_POSITIVE":
        # Buscar en la traza la última derivación responsable
        for entry in reversed(trace):
            if entry.get("type") == "derivation":
                rule_id = entry.get("rule_id")
                if rule_id and rule_id in theory.rules:
                    return theory.rules[rule_id]

        # Fallback: primera regla del predicado target
        rules = theory.get_rules_for_predicate(target_predicate)
        if rules:
            return rules[0]

    elif error_type == "FALSE_NEGATIVE":
        rules = theory.get_rules_for_predicate(target_predicate)
        # CORTEX-OMEGA: Only consider POSITIVE rules for False Negatives.
        # If we pick a negative rule, we'll just branch it into another negative rule,
        # which doesn't solve the missing positive coverage.
        positive_rules = [r for r in rules if not r.head.negated]
        
        if positive_rules:
            return positive_rules[0]
        # Fallback for Cold Start: If no rules exist, we can't identify a culprit.
        # But we still need to generate a candidate.
        # We return None here, but generate_structural_candidates needs to handle it.
        return None

    return None
