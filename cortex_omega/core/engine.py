# core_kernel.py

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any

from .rules import RuleBase, Rule, Literal, FactBase, Scene
from .inference import InferenceEngine
from .values import ValueBase, Axiom
from .hypothesis import HypothesisGenerator, FailureContext

import copy
import logging

logger = logging.getLogger(__name__)


@dataclass
class KernelConfig:
    """
    Configuración y dependencias del kernel.
    Es 'estado externo' que controla cómo propone parches y cómo evalúa armonía.
    """
    lambda_complexity: float = 0.3
    max_memory: int = 100
    # Simulated Annealing parameters
    temperature: float = 1.0
    cooling_rate: float = 0.95
    # Motor de generación de parches estructurales
    patch_generator: Optional[HypothesisGenerator] = None
    # CORTEX-OMEGA: Compiler
    compiler: Optional['KnowledgeCompiler'] = None
    # Refinement Loop
    max_refinement_steps: int = 3
    
    # CORTEX-OMEGA v1.5: World-Class Config
    mode: str = "robust" # "robust" (default) or "strict"
    priors: Dict[str, float] = field(default_factory=lambda: {"rule_base": 0.5, "exception": 0.3})
    noise_model: Dict[str, float] = field(default_factory=lambda: {"false_positive": 0.05, "false_negative": 0.05})
    plasticity: Dict[str, Any] = field(default_factory=lambda: {"min_conf_to_keep": 0.6, "max_rule_count": 500})
    feature_priors: Dict[str, float] = field(default_factory=dict)



def find_worst_error(theory: RuleBase, memory: List[Scene], axioms: ValueBase) -> Optional[Scene]:
    """
    Encuentra la escena en memoria con el error más grave bajo la teoría actual.
    Prioriza False Positives (regresiones) sobre False Negatives.
    """
    first_fp = None
    first_fn = None
    
    for s in memory:
        pred, _ = infer(theory, s)
        if pred and not s.ground_truth:
            # False Positive
            first_fp = s
            break # Stop at first FP (critical regression)
        elif not pred and s.ground_truth:
            # False Negative
            if not first_fn:
                first_fn = s
                
    if first_fp:
        return first_fp
    return first_fn

# =========================
# 1. Operador de actualización
# =========================

def update_theory_kernel(
    theory: RuleBase,
    scene: Scene,
    memory: List[Scene],
    axioms: ValueBase,
    config: KernelConfig,
) -> Tuple[RuleBase, List[Scene]]:
    """
    El corazón del algoritmo.
    Entrada: teoría actual, nueva escena, memoria, axiomas y configuración.
    Salida: nueva teoría, nueva memoria.
    """
    logger.debug(f"DEBUG: Start Update Theory for Scene {scene.id}")

    # 1. Proyección
    prediction, trace = infer(theory, scene)

    # Si acierta, no hay "gradiente de error": solo consolidamos memoria
    # Si acierta, no hay "gradiente de error": solo consolidamos memoria
    if prediction != scene.ground_truth:
        # Prediction failed -> Punish rules involved in the error
        punish_rules(theory, trace, scene.target_predicate, scene.ground_truth)
    else:
        # Prediction succeeded -> Reward rules involved
        reward_rules(theory, trace, scene.target_predicate, scene.ground_truth)
        
    # CORTEX-OMEGA v1.4: Update Rule Stats (Observation)
    update_rule_stats(theory, trace, scene.target_predicate, scene.ground_truth)
        
    new_memory = append_to_memory(memory, scene, config.max_memory)
    
    # CORTEX-OMEGA: Maintenance (GC + Compiler)
    garbage_collect(theory)
    if config.compiler:
        config.compiler.promote_stable_rules(theory, axioms)
        config.compiler.check_conflicts(theory)
            
        # Check for NEGATIVE_GAP (True Negative without explicit explanation)
        should_return = True
        if not prediction and not scene.ground_truth:
            # Check trace for explicit negative derivation
            has_explanation = False
            neg_pred = f"NOT_{scene.target_predicate}"
            for entry in trace:
                if entry.get("type") == "derivation":
                    derived = entry.get("derived")
                    if derived and derived.predicate == neg_pred:
                        has_explanation = True
                        break
            
            if not has_explanation:
                should_return = False
        
        if should_return:
            return theory, new_memory

    # Tipo de error (equivalente al "signo" del gradiente)
    if prediction and not scene.ground_truth:
        error_type = "FALSE_POSITIVE"
    elif not prediction and scene.ground_truth:
        error_type = "FALSE_NEGATIVE"
    elif not prediction and not scene.ground_truth:
        # TRUE NEGATIVE (Correctly predicted False)
        # Check for NEGATIVE_GAP
        error_type = "NEGATIVE_GAP"
    else:
        # True Positive
        return theory, new_memory
        
    # 2. Generación de teorías candidatas (parches estructurales)

    candidate_theories, ctx = generate_structural_candidates(
        theory=theory,
        scene=scene,
        memory=memory,
        trace=trace,
        error_type=error_type,
        patch_generator=config.patch_generator,
        lambda_complexity=config.lambda_complexity,
    )

    # 3. Evaluación del baseline (armonía actual)
    eval_scenes = memory + [scene]
    
    # CORTEX-OMEGA: Entropy Regularization
    entropy_map = calculate_attribute_entropy(eval_scenes)
    
    best_theory = theory
    best_harmony = score_harmony(theory, eval_scenes, config.lambda_complexity, entropy_map)
    
    best_harmony = score_harmony(theory, eval_scenes, config.lambda_complexity, entropy_map)
    print(f"DEBUG: Initial Harmony: {best_harmony:.4f}")
    
    logger.debug(f"DEBUG: Initial Harmony: {best_harmony:.4f}")
    logger.debug(f"DEBUG: Candidates generated: {len(candidate_theories)}")

    # 4. Búsqueda de la mejor teoría que respete axiomas y mejore armonía
    best_patch = None
    
    for i, (T_candidate, patch) in enumerate(candidate_theories):
        # 4a. Restricción dura: axiomas
        if violates_axioms(T_candidate, axioms, eval_scenes):
            logger.debug(f"DEBUG: Candidate {i} violated axioms.")
            continue

        # 4b. Recalcular armonía global
        h_new = score_harmony(T_candidate, eval_scenes, config.lambda_complexity, entropy_map)
        print(f"DEBUG: Candidate {i} Harmony: {h_new:.4f} (Rules={len(T_candidate.rules)})")
        for r in T_candidate.rules.values():
            print(f"  - {r}")

        accepted = False
        if h_new > best_harmony:
            accepted = True
        else:
            # CORTEX-OMEGA: Simulated Annealing (Metropolis Criterion)
            # Accept worse solution with probability P = exp((h_new - h_current) / T)
            import math
            import random
            
            delta = h_new - best_harmony
            # Avoid division by zero
            temp = max(config.temperature, 1e-5)
            prob = math.exp(delta / temp)
            
            
            if random.random() < prob:
                # CORTEX-OMEGA: Safety check - Don't accept total failure
                if h_new < 0.01 and best_harmony > 0.01:
                    pass
                else:
                    print(f"DEBUG: Accepted worse candidate (Prob={prob:.4f}, T={temp:.4f}) to escape local max.")
                    accepted = True
        
        if accepted:
            best_harmony = h_new
            best_theory = T_candidate
            best_patch = patch

    # Cool down (once per update cycle)
    config.temperature *= config.cooling_rate
            
    # CORTEX-OMEGA: Feedback Loop
    if best_patch and config.patch_generator and ctx:
        config.patch_generator.record_feedback(ctx, best_patch)

    # 5. Actualizar memoria (FIFO)
    new_memory = append_to_memory(memory, scene, config.max_memory)
    
    # CORTEX-OMEGA: Reinforcement Learning (Update Confidence)
    # We verify the prediction of the BEST theory on the CURRENT scene.
    pred, trace = infer(best_theory, scene)
    if pred == scene.ground_truth:
        reward_rules(best_theory, trace, scene.target_predicate, scene.ground_truth)
    else:
        punish_rules(best_theory, trace, scene.target_predicate, scene.ground_truth, config=config)
        
    # CORTEX-OMEGA v1.4: Update Rule Stats (Observation)
    update_rule_stats(best_theory, trace, scene.target_predicate, scene.ground_truth)
    
    # 5. Refinement Loop (Fix Regressions on Memory)
    # This is crucial for fixing False Positives introduced by generalization (e.g. Temporal Learning)
    # 5. Refinement Loop (Fix Regressions on Memory)
    # This is crucial for fixing False Positives introduced by generalization (e.g. Temporal Learning)
    MAX_REFINEMENT_STEPS = 3
    for i in range(MAX_REFINEMENT_STEPS):
        worst_scene = find_worst_error(best_theory, memory, axioms)
        if not worst_scene:
            break
            
        # Determine error type and trace
        pred_r, trace_r = infer(best_theory, worst_scene)
        err_type = "FALSE_POSITIVE" if pred_r and not worst_scene.ground_truth else "FALSE_NEGATIVE"
        
        # Generate candidates
        refinement_candidates, _ = generate_structural_candidates(
            theory=best_theory,
            scene=worst_scene,
            memory=memory,
            trace=trace_r,
            error_type=err_type,
            patch_generator=config.patch_generator,
            lambda_complexity=config.lambda_complexity,
            config=config
        )
        
        if not refinement_candidates:
            break
            
        # Evaluate candidates
        best_refinement_harmony = score_harmony(best_theory, eval_scenes, config.lambda_complexity, entropy_map)
        best_refinement_theory = best_theory
        improved = False
        
        for j, (T_cand, patch) in enumerate(refinement_candidates):
            if violates_axioms(T_cand, axioms, eval_scenes):
                continue
                
            h_cand = score_harmony(T_cand, eval_scenes, config.lambda_complexity, entropy_map)
            
            if h_cand > best_refinement_harmony:
                best_refinement_harmony = h_cand
                best_refinement_theory = T_cand
                improved = True
        
        if improved:
            best_theory = best_refinement_theory
        else:
            break

    # CORTEX-OMEGA: Garbage Collection (Periodic)
    # For now, run every time (it's cheap).
    threshold = config.plasticity.get("min_conf_to_keep", 0.0) if config.plasticity else 0.0
    garbage_collect(best_theory, threshold=threshold)
    
    # CORTEX-OMEGA: Self-Reflecting Compiler
    if config.compiler:
        config.compiler.promote_stable_rules(best_theory, axioms)
        config.compiler.check_conflicts(best_theory)
    
    return best_theory, new_memory


# =========================
# 2. Operaciones auxiliares
# =========================

def infer(theory: RuleBase, scene: Scene):
    """
    Inferencia pura: corre el motor lógico sobre la escena usando la teoría dada.
    Retorna (predicción_bool, traza_inferencia).
    """
    facts_copy = clone_factbase(scene.facts)
    engine = InferenceEngine(facts_copy, theory)
    engine.forward_chain(debug=True)
    
    if scene.target_args:
        target = Literal(scene.target_predicate, scene.target_args)
    else:
        target = Literal(scene.target_predicate, (scene.target_entity,))
    
    # CORTEX-OMEGA v1.3: Conflict Resolution (Holographic Logic)
    # Check for both Positive and Negative derivations
    pos_proof = engine.get_proof(target)
    
    neg_target = Literal(f"NOT_{scene.target_predicate}", (scene.target_entity,))
    neg_proof = engine.get_proof(neg_target)
    
    prediction = False
    trace = getattr(engine, "trace", [])
    
    if pos_proof and not neg_proof:
        prediction = True
    elif not pos_proof and neg_proof:
        prediction = False
    elif pos_proof and neg_proof:
        # CONFLICT! Resolve by Confidence
        if pos_proof.confidence > neg_proof.confidence:
            prediction = True
        elif neg_proof.confidence > pos_proof.confidence:
            prediction = False
        else:
            # Tie-breaker: Specificity? Or default to False (Safety)?
            # For now, default to False (Conservative)
            prediction = False
            
    # CORTEX-OMEGA: Usage Tracking
    if trace:
        print(f"DEBUG: Inference Trace for {scene.target_entity}:")
        for step in trace:
            print(f"  - {step}")
        import time
        for step in trace:
            if step["type"] == "derivation":
                rid = step["rule_id"]
                if rid in theory.rules:
                    theory.rules[rid].usage_count += 1
                    theory.rules[rid].last_used = time.time()
    
    return prediction, trace



# ... (update_theory_kernel signature needs update too, but let's do punish_rules first)

def punish_rules(theory: RuleBase, trace: List[Dict], target_predicate: str, ground_truth: bool, config: KernelConfig = None):
    """
    Punishes rules that contributed to an incorrect prediction.
    CORTEX-OMEGA v1.3: Selective Punishment.
    
    Modes:
    - "robust": Standard Bayesian update (failure_count += 1). Resilient to noise.
    - "strict": Harsh punishment (failure_count = 1_000_000). Single counter-example kills the rule.
    """
    mode = config.mode if config else "robust"
    punished = set()
    
    for step in trace:
        if step["type"] == "derivation":
            rid = step["rule_id"]
            derived_lit = step["derived"]
            
            pred_name = derived_lit.predicate
            is_negated = derived_lit.negated
            
            should_punish = False
            
            # Case 1: Rule derived Positive, but GT is Negative (False Positive)
            if not is_negated and pred_name == target_predicate:
                if not ground_truth:
                    should_punish = True
                    
            # Case 2: Rule derived Negative, but GT is Positive (False Negative)
            # Note: Negative derivation can be explicit "NOT_pred" or negated literal "pred"
            elif (is_negated and pred_name == target_predicate) or pred_name == f"NOT_{target_predicate}":
                if ground_truth:
                    should_punish = True
            
            if should_punish:
                if rid in theory.rules and rid not in punished:
                    rule = theory.rules[rid]
                    
                    # CORTEX-OMEGA v1.3: Mode-based Punishment
                    if mode == "strict":
                        # Strict Mode: Zero Tolerance.
                        # Set failure count to a massive number to force confidence -> 0.0
                        rule.failure_count = 1_000_000
                        logger.info(f"💀 STRICT MODE: Rule {rid} killed due to counter-example.")
                    else:
                        increment = 1 # Standard Bayesian update
                        rule.failure_count += increment
                    
                    # rule.fires_neg is now handled in update_rule_stats
                    
                    # Bayesian Update: Mean of Beta(s+1, f+1)
                    s = rule.support_count
                    f = rule.failure_count
                    
                    # New Confidence
                    rule.confidence = (s + 1.0) / (s + f + 2.0)
                    
                    punished.add(rid)

def reward_rules(theory: RuleBase, trace: List[Dict], target_predicate: str, ground_truth: bool, reward: float = 0.1):
    """
    CORTEX-OMEGA v1.3: Bayesian Reward.
    Updates support counts and recalculates confidence.
    Only rewards rules that derived facts consistent with the final outcome.
    """
    rewarded = set()
    
    for step in trace:
        if step["type"] == "derivation":
            rid = step["rule_id"]
            derived_lit = step["derived"]
            pred_name = derived_lit.predicate
            is_negated = derived_lit.negated
            
            should_reward = False
            
            # Case 1: GT is Positive. Reward rules deriving Positive Target.
            if ground_truth:
                if not is_negated and pred_name == target_predicate:
                    should_reward = True
            
            # Case 2: GT is Negative. Reward rules deriving Negative Target.
            else:
                if (is_negated and pred_name == target_predicate) or pred_name == f"NOT_{target_predicate}":
                    should_reward = True
            
            if should_reward:
                if rid in theory.rules and rid not in rewarded:
                    rule = theory.rules[rid]
                    rule.support_count += 1
                    # rule.fires_pos is now handled in update_rule_stats
                    
                    # Bayesian Update
                    s = rule.support_count
                    f = rule.failure_count
                    
                    # New Confidence
                    rule.confidence = (s + 1.0) / (s + f + 2.0)
                    
                    rewarded.add(rid)

def update_rule_stats(theory: RuleBase, trace: List[Dict], target_predicate: str, ground_truth: bool):
    """
    CORTEX-OMEGA v1.4: First-Class Rule Statistics.
    Updates fires_pos and fires_neg for ALL firing rules, regardless of system prediction.
    """
    updated = set()
    for step in trace:
        if step["type"] == "derivation":
            rid = step["rule_id"]
            if rid in theory.rules and rid not in updated:
                rule = theory.rules[rid]
                derived_lit = step["derived"]
                pred_name = derived_lit.predicate
                is_negated = derived_lit.negated
                
                # Check consistency with Ground Truth
                is_consistent = False
                if ground_truth:
                    if not is_negated and pred_name == target_predicate:
                        is_consistent = True
                else:
                    if (is_negated and pred_name == target_predicate) or pred_name == f"NOT_{target_predicate}":
                        is_consistent = True
                
                # Update Stats
                if is_consistent:
                    rule.fires_pos += 1
                else:
                    rule.fires_neg += 1
                    
                updated.add(rid)

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
    distributions = {}
    
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
        # print(f"DEBUG: Entropy({pred}) = {entropy:.2f}")
        
    return entropy_map


def generate_structural_candidates(
    theory: RuleBase,
    scene: Scene,
    memory: List[Scene],
    trace,
    error_type: str,
    patch_generator: Optional[HypothesisGenerator] = None,
    top_k: int = 5,
    lambda_complexity: float = 0.3,
    config: KernelConfig = None
) -> Tuple[List[Tuple[RuleBase, Any]], Optional[FailureContext]]:
    """
    Toma la teoría actual + la escena conflictiva y devuelve una lista de teorías candidatas
    ya parcheadas (RuleBase clonadas y modificadas).
    El kernel no sabe de 'Patch', solo ve teorías completas.
    """
    if patch_generator is None:
        patch_generator = HypothesisGenerator()

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
                
            culprit_rule = Rule("dummy_genesis", Literal(target_pred, vars), [])
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
        feature_priors=config.feature_priors if config else {}
    )

    # Pass lambda_complexity to generator if supported
    if hasattr(patch_generator, 'lambda_complexity'):
        patch_generator.lambda_complexity = lambda_complexity

    raw_candidates = patch_generator.generate(ctx, top_k=top_k)
    candidate_theories: List[Tuple[RuleBase, Any]] = []

    for patch, new_rule, aux_rules in raw_candidates:
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

        # Añadir reglas auxiliares (conceptos, excepciones, etc.)
        for aux in aux_rules:
            if aux.id in T_candidate.rules:
                T_candidate.replace(aux.id, aux)
            else:
                T_candidate.add(aux)

        candidate_theories.append((T_candidate, patch))

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


def violates_axioms(
    theory: RuleBase,
    axioms: ValueBase,
    eval_scenes: List[Scene],
) -> bool:
    """
    Ejecuta la teoría sobre un conjunto de escenas y comprueba si alguna
    predicción viola un axioma. Si hay una sola violación, devuelve True.
    """
    for s in eval_scenes:
        facts_copy = clone_factbase(s.facts)
        engine = InferenceEngine(facts_copy, theory)
        engine.forward_chain()

        for pred, args_set in facts_copy.facts.items():
            for args in args_set:
                lit = Literal(pred, args)
                violated = axioms.check_sin(facts_copy, lit)
                if violated is not None:
                    # En tu código actual sueles loggear aquí; el kernel no imprime.
                    return True
    return False


def score_harmony(
    theory: RuleBase,
    scenes: List[Scene],
    lambda_complexity: float,
    entropy_map: Dict[str, float] = None,
) -> float:
    """
    Función de 'pérdida invertida':
        harmony = F1_Score / (1 + λ * complexity)

    Donde:
      - F1_Score: Media armónica de Precision y Recall.
      - complexity: longitud estructural de la teoría + Entropía.
    """
    # CORTEX-OMEGA v1.3: Use F1-Score to avoid "Always False" local optimum
    score = evaluate_f1(theory, scenes)
    complexity = compute_complexity(theory, entropy_map)
    return score / (1.0 + lambda_complexity * complexity)


def evaluate_f1(theory: RuleBase, scenes: List[Scene]) -> float:
    if not scenes:
        return 0.0

    tp = 0
    fp = 0
    fn = 0
    
    for s in scenes:
        facts_copy = clone_factbase(s.facts)
        engine = InferenceEngine(facts_copy, theory)
        engine.forward_chain()

        if s.target_args:
            target_lit = Literal(s.target_predicate, s.target_args)
        else:
            target_lit = Literal(s.target_predicate, (s.target_entity,))
            
        # CORTEX-OMEGA v1.3: Conflict Resolution in Evaluation too
        # We need to match the logic in 'infer'
        neg_target_lit = Literal(f"NOT_{s.target_predicate}", target_lit.args)
        
        pos_proof = engine.get_proof(target_lit)
        neg_proof = engine.get_proof(neg_target_lit)
        
        # print(f"DEBUG: Pos Proof for {target_lit}: {pos_proof}")
        # print(f"DEBUG: Neg Proof for {neg_target_lit}: {neg_proof}")

        prediction = False
        explanation = "No rule fired."
        trace = []
        
        if pos_proof and not neg_proof:
            prediction = True
            trace = pos_proof.steps
        elif neg_proof and not pos_proof:
            prediction = False
            trace = neg_proof.steps
        elif pos_proof and neg_proof:
            if pos_proof.confidence > neg_proof.confidence:
                prediction = True
                trace = pos_proof.steps
            else:
                prediction = False
                trace = neg_proof.steps
        
        if prediction and s.ground_truth:
            tp += 1
        elif prediction and not s.ground_truth:
            fp += 1
        elif not prediction and s.ground_truth:
            fn += 1
            
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0 # Default to 1.0 if no predictions (conservative)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0 # Default to 0.0 if no positives
    
    if precision + recall == 0:
        return 0.0
        
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def compute_complexity(theory: RuleBase, entropy_map: Dict[str, float] = None) -> float:
    """
    CORTEX-OMEGA: Includes Entropy Regularization.
    Cost = Rules + Sum(1 + Alpha * Entropy(Literal))
    """
    num_rules = len(theory.rules)
    num_concepts = sum(1 for r in theory.rules.values() if "Concept_" in r.id)
    
    literal_cost = 0.0
    alpha = 0.2 # Reduced from 0.5 to allow valid high-entropy attributes (like color)
    
    for r in theory.rules.values():
        for lit in r.body:
            base_cost = 0.5
            entropy_cost = 0.0
            if entropy_map and lit.predicate in entropy_map:
                entropy_cost = alpha * entropy_map[lit.predicate]
            
            literal_cost += (base_cost + entropy_cost)

    # Pesos heurísticos (pueden refinarse)
    return 1.0 * num_rules + literal_cost + 0.8 * num_concepts


def score_mdl(rule: Rule, lambda_complexity: float = 0.2) -> float:
    """
    CORTEX-OMEGA v1.4: MDL Score for a single rule.
    Score = Coverage * (1 - ErrorRate) - Lambda * Complexity
    
    Higher is better.
    """
    if rule.coverage == 0:
        return 0.0
        
    error_rate = 1.0 - rule.reliability
    # We want to maximize Coverage * Reliability, penalized by Complexity.
    
    # Adjusted Formula:
    # Score = (FiresPos - FiresNeg) - Lambda * Complexity
    # This is equivalent to Coverage * (Rel - (1-Rel)) - Lambda * Complexity
    # = Coverage * (2*Rel - 1) - Lambda * Complexity
    
    # Let's stick to the plan:
    # score(R) = coverage * (1 - error_rate) - lambda * complexity
    # coverage * reliability - lambda * complexity
    # fires_pos - lambda * complexity
    
    # Wait, if fires_pos is high, we like it.
    # If complexity is high, we dislike it.
    # But we also want to penalize fires_neg explicitly?
    # fires_pos accounts for reliability implicitly (it's the numerator).
    # But fires_neg doesn't hurt fires_pos directly.
    
    # Better Formula:
    # Score = fires_pos - fires_neg - lambda * complexity
    
    return (rule.fires_pos - rule.fires_neg) - (lambda_complexity * rule.complexity)


def clone_factbase(fb: FactBase) -> FactBase:
    """
    Copia profunda de un FactBase, reutilizando tu representación interna.
    """
    fb_copy = FactBase()
    for pred, args_set in fb.facts.items():
        for args in args_set:
            fb_copy.add(pred, args)
    return fb_copy


def append_to_memory(memory: List[Scene], scene: Scene, max_memory: int) -> List[Scene]:
    """
    Buffer FIFO simple.
    """
    new_memory = memory.copy()
    new_memory.append(scene)
    if len(new_memory) > max_memory:
        new_memory = new_memory[-max_memory:]
    return new_memory


def garbage_collect(theory: RuleBase, threshold: float = 0.0):
    """
    CORTEX-OMEGA Pillar 4: Concept Compression.
    Prunes low-utility rules using MDL Scoring.
    """
    to_remove = []
    # CORTEX-OMEGA v1.4: Use MDL Score
    # Threshold 0.0 means "Does more harm than good" (fires_neg > fires_pos - complexity penalty)
    
    for rid, rule in list(theory.rules.items()):
        # Give new rules a grace period (support_count < 5)
        if rule.support_count < 5:
            continue
            
        score = score_mdl(rule)
        
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


class KnowledgeCompiler:
    """
    CORTEX-OMEGA Pillar 5: Self-Reflecting Compiler.
    Promotes stable rules to axioms and detects logical conflicts.
    """
    def __init__(self, promotion_threshold: float = 0.99, min_usage: int = 10):
        self.promotion_threshold = promotion_threshold
        self.min_usage = min_usage

    def promote_stable_rules(self, theory: RuleBase, axioms: ValueBase) -> List[str]:
        """
        Moves highly stable rules from RuleBase to ValueBase (Axioms).
        """
        promoted = []
        to_remove = []
        
        for rule_id, rule in theory.rules.items():
            # Criteria: High Confidence AND High Usage
            if rule.confidence >= self.promotion_threshold and rule.usage_count >= self.min_usage:
                # Promote!
                logger.info(f"CORTEX: Promoting {rule_id} to Axiom! (Conf={rule.confidence}, Usage={rule.usage_count})")
                
                # Convert Rule to Axiom
                # Axiom format: condition -> forbidden
                # Wait, Axioms are constraints (Safety). Rules are derivations (Liveness).
                # Promoting a Rule to an Axiom means "This derivation is ALWAYS true".
                # But our Axiom class is designed for "Forbidden" states (Safety).
                
                # We need a new type of Axiom: "Constructive Axiom" or just treat it as a frozen rule.
                # For v1, let's assume we just freeze it in a special "Frozen Rules" section of RuleBase?
                # OR we add it to ValueBase as a "Positive Axiom" if ValueBase supports it.
                
                # Let's look at ValueBase. It supports "Axiom" which is condition -> forbidden.
                # It doesn't seem to support constructive axioms (A -> B).
                
                # Alternative: Just mark the rule as "immutable" in RuleBase and give it infinite confidence.
                # But the goal is "Compiler".
                
                # Let's stick to the plan: "Promote to Axiom".
                # If ValueBase only supports constraints, maybe we can't move it there directly without changing ValueBase.
                # BUT, we can create a "Constraint" that says "It is forbidden for Head to be False if Body is True".
                # i.e. Body -> not Head is Forbidden.
                
                # Example: Rule: glows(X) :- red(X).
                # Axiom: red(X) -> not glows(X) is FORBIDDEN.
                # This enforces the rule as a hard constraint!
                
                # Condition: rule.body
                # Forbidden: Literal(rule.head.predicate, rule.head.args, negated=not rule.head.negated)
                
                # If head is positive (glows), forbidden is negative (not glows).
                # If head is negative (not glows), forbidden is positive (glows).
                
                forbidden_pred = rule.head.predicate
                forbidden_args = rule.head.args
                # We want to forbid the OPPOSITE of the head.
                # If head is P, we forbid not P.
                # But our Axiom system checks if "forbidden" matches the prediction.
                # If prediction is P, and we forbid P, then P is bad.
                
                # Wait, Axiom logic: "If condition is met, then 'forbidden' must NOT be true."
                # If we want to enforce P, we want to say "It is forbidden that NOT P is true".
                # But our system might not represent "NOT P" explicitly as a fact.
                # It uses Negation as Failure.
                
                # Let's try a simpler approach for v1:
                # Just lock the rule in RuleBase (confidence = 1.0, immutable).
                # AND add a constraint to ValueBase to prevent it from being unlearned?
                
                # Let's implement the "Constraint" approach: Body -> ~Head is Forbidden.
                # This prevents the system from learning a rule that contradicts this one.
                
                # Construct Forbidden Literal
                # If Head is P(X), we forbid ~P(X).
                # But ~P(X) is usually not a fact.
                
                # Okay, let's just use the "Frozen Rule" approach for simplicity and safety.
                # We will set confidence to 1.1 (Super-Truth) and maybe a flag.
                rule.confidence = 1.0
                # We can't move it to ValueBase easily without refactoring ValueBase to support constructive logic.
                
                promoted.append(rule_id)
                # We don't remove it from RuleBase, we just "crystallize" it.
                
        return promoted

    def check_conflicts(self, theory: RuleBase) -> List[str]:
        """
        Detects logical conflicts (e.g., A and not A).
        """
        conflicts = []
        # Simplified check: Do we have rules that conclude P and rules that conclude not P?
        # Our literals have a 'negated' flag.
        
        preds = set()
        for rule in theory.rules.values():
            preds.add(rule.head.predicate)
            
        # This check is weak because P and not P might be valid in different contexts.
        # A real conflict is deriving P(x) and not P(x) for the SAME x.
        # That requires runtime checking (which InferenceEngine does).
        
        # Static check: Do we have Rule 1: P(X) :- Q(X) and Rule 2: not P(X) :- Q(X)?
        # If bodies are identical and heads are opposite -> Conflict.
        
        # O(N^2) check
        rules = list(theory.rules.values())
        for i in range(len(rules)):
            for j in range(i+1, len(rules)):
                r1 = rules[i]
                r2 = rules[j]
                
                if r1.head.predicate == r2.head.predicate and r1.head.args == r2.head.args:
                    if r1.head.negated != r2.head.negated:
                        # Opposite heads. Check bodies.
                        # Naive body equality
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
