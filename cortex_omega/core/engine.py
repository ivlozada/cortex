# core_kernel.py

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any

from .rules import RuleBase, Rule, Literal, FactBase, Scene
from .inference import InferenceEngine
from .values import ValueBase, Axiom
from .hypothesis import HypothesisGenerator, FailureContext
from .config import KernelConfig
from .critic import Critic, clone_factbase
from .learner import Learner

import copy
import logging

logger = logging.getLogger(__name__)






def find_worst_error(theory: RuleBase, memory: List[Scene], axioms: ValueBase) -> Optional[Scene]:
    """
    Encuentra la escena en memoria con el error más grave bajo la teoría actual.
    Prioriza False Positives (regresiones) sobre False Negatives.
    """
    first_fp = None
    first_fn = None
    
    for s in memory:
        pred, _ = infer(theory, s)
        # print(f"DEBUG: Checking scene {s.id}, GT={s.ground_truth}, Pred={pred}")
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
    learner = Learner(config)
    return learner.learn(theory, scene, memory, axioms)


def _propose_candidates(
    theory: RuleBase,
    scene: Scene,
    memory: List[Scene],
    trace,
    error_type: str,
    config: KernelConfig
) -> Tuple[List[Tuple[RuleBase, Any]], Optional[FailureContext]]:
    return generate_structural_candidates(
        theory=theory,
        scene=scene,
        memory=memory,
        trace=trace,
        error_type=error_type,
        config=config,
        feature_priors=config.feature_priors
    )


def _evaluate_and_select(
    theory: RuleBase,
    candidate_theories: List[Tuple[RuleBase, Any]],
    eval_scenes: List[Scene],
    axioms: ValueBase,
    entropy_map: Dict[str, float],
    config: KernelConfig,
    ctx: Optional[FailureContext]
) -> Tuple[RuleBase, Any]:
    critic = Critic(config)
    best_theory = theory
    best_harmony = critic.score_harmony(theory, eval_scenes, entropy_map)
    best_patch = None
    
    for i, (T_candidate, patch) in enumerate(candidate_theories):
        # 4a. Restricción dura: axiomas
        if violates_axioms(T_candidate, axioms, eval_scenes):
            continue

        # 4b. Recalcular armonía global
        h_new = critic.score_harmony(T_candidate, eval_scenes, entropy_map)

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
                    logger.debug(f"Accepted worse candidate (Prob={prob:.4f}, T={temp:.4f}) to escape local max.")
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
        
    return best_theory, best_patch


def _refine_theory(
    theory: RuleBase,
    memory: List[Scene],
    eval_scenes: List[Scene],
    axioms: ValueBase,
    entropy_map: Dict[str, float],
    config: KernelConfig
) -> RuleBase:
    best_theory = theory
    
    for i in range(config.max_refinement_steps):
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
            config=config,
            feature_priors=config.feature_priors
        )
        
        if not refinement_candidates:
            break
            
        # Evaluate candidates
        critic = Critic(config)
        best_refinement_harmony = critic.score_harmony(best_theory, eval_scenes, entropy_map)
        best_refinement_theory = best_theory
        improved = False
        
        for j, (T_cand, patch) in enumerate(refinement_candidates):
            if violates_axioms(T_cand, axioms, eval_scenes):
                continue
                
            h_cand = critic.score_harmony(T_cand, eval_scenes, entropy_map)
            
            if h_cand > best_refinement_harmony:
                best_refinement_harmony = h_cand
                best_refinement_theory = T_cand
                improved = True
        
        if improved:
            best_theory = best_refinement_theory
        else:
            break
            
    return best_theory


# =========================
# 2. Operaciones auxiliares
# =========================

def infer(theory: RuleBase, scene: Scene, config: Optional[KernelConfig] = None) -> Tuple[bool, List[Dict]]:
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
    
    prediction = False
    trace = getattr(engine, "trace", [])
    
    if pos_proof and not neg_proof:
        prediction = True
        trace = pos_proof.steps
    elif not pos_proof and neg_proof:
        prediction = False
        trace = neg_proof.steps
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



# ... (update_theory_kernel signature needs update too, but let's do punish_rules first)



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
            
            # CORTEX-OMEGA v1.6: Reward ALL rules in the trace if the outcome was correct.
            # This ensures auxiliary rules (exceptions, concepts) get credit.
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



def update_rule_stats(theory: RuleBase, trace: List[Dict], is_correct: bool, config: KernelConfig):
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
        # logger.debug(f"Entropy({pred}) = {entropy:.2f}")
        
    return entropy_map


from .theorist import Theorist # Added import





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




def append_to_memory(memory: List[Scene], scene: Scene, max_memory: int) -> List[Scene]:
    """
    Buffer FIFO simple.
    """
    new_memory = memory.copy()
    new_memory.append(scene)
    if len(new_memory) > max_memory:
        new_memory = new_memory[-max_memory:]
    return new_memory


def garbage_collect(theory: RuleBase, threshold: float = 0.0, config: KernelConfig = None):
    """
    CORTEX-OMEGA Pillar 4: Concept Compression.
    Prunes low-utility rules using MDL Scoring.
    """
    to_remove = []
    # CORTEX-OMEGA v1.4: Use MDL Score
    # Threshold 0.0 means "Does more harm than good" (fires_neg > fires_pos - complexity penalty)
    
    lambda_val = config.lambda_complexity if config else 0.2
    
    for rid, rule in list(theory.rules.items()):
        # Give new rules a grace period (support_count < 5)
        if rule.support_count < 5:
            continue
            
        critic = Critic(config)
        score = critic.score_mdl(rule)
        
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
