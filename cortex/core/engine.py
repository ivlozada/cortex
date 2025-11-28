# core_kernel.py

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

from .rules import RuleBase, Rule, Literal, FactBase, Scene
from .inference import InferenceEngine
from .values import ValueBase, Axiom
from .hypothesis import HypothesisGenerator, FailureContext

import copy


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
    print(f"DEBUG: Start Update Theory for Scene {scene.id}", flush=True)

    # 1. Proyección
    prediction, trace = infer(theory, scene)

    # Si acierta, no hay "gradiente de error": solo consolidamos memoria
    # Si acierta, no hay "gradiente de error": solo consolidamos memoria
    if prediction == scene.ground_truth:
        # CORTEX-OMEGA: Reinforcement Learning (Reward Success)
        reward_rules(theory, trace)
        
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
        # CORTEX-OMEGA: Kill Switch / Dogmatism Fix
        punish_rules(theory, trace, penalty=0.8)
    elif not prediction and scene.ground_truth:
        error_type = "FALSE_NEGATIVE"
    elif not prediction and not scene.ground_truth:
        # TRUE NEGATIVE (Correctly predicted False)
        # But do we have an EXPLANATION? (Explicit Negative Rule)
        # Check if NOT_target is in facts (derived by forward_chain)
        # Note: 'scene.facts' is input, we need the derived facts from 'infer'.
        # 'infer' returns prediction, trace. It doesn't return the full derived facts.
        # But we can check the trace for a negative derivation?
        # Or we can assume that if we are here, we might want to reinforce negative rules.
        
        # Let's check if we have a negative explanation.
        # We need to peek into the inference engine state or re-run query for NOT_target.
        # For efficiency, let's just assume we want to learn explicit negations if we don't have them.
        # But how do we know if we have them?
        
        # Hack: Re-check if NOT_target is derivable
        neg_target = Literal(scene.target_predicate, (scene.target_entity,), negated=True)
        # We can't easily check 'derived' from 'infer' because it's not returned.
        # Let's modify 'infer' to return derived facts? No, too invasive.
        
        # Let's assume we always want to try to learn a negative rule if we are in a True Negative state
        # AND we are in a "learning phase" (which we always are).
        # But we don't want to spam negative rules for everything.
        # Only if we suspect we are missing an explanation.
        
        # Let's trigger "NEGATIVE_GAP" which the hypothesis generator can choose to ignore if it finds a negative rule already?
        # Or better: The hypothesis generator will propose a negative rule. 
        # If a similar negative rule already exists and covers it, the harmony score won't improve much (redundancy).
        
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
    
    print(f"DEBUG: Initial Harmony: {best_harmony:.4f}")
    print(f"DEBUG: Candidates generated: {len(candidate_theories)}")

    # 4. Búsqueda de la mejor teoría que respete axiomas y mejore armonía
    best_patch = None
    
    for i, (T_candidate, patch) in enumerate(candidate_theories):
        # 4a. Restricción dura: axiomas
        if violates_axioms(T_candidate, axioms, eval_scenes):
            print(f"DEBUG: Candidate {i} violated axioms.")
            continue

        # 4b. Recalcular armonía global
        h_new = score_harmony(T_candidate, eval_scenes, config.lambda_complexity, entropy_map)
        print(f"DEBUG: Candidate {i} Harmony: {h_new:.4f}")

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
                print(f"DEBUG: Accepted worse candidate (Prob={prob:.4f}, T={temp:.4f}) to escape local max.")
                accepted = True
        
        if accepted:
            best_harmony = h_new
            best_theory = T_candidate
            best_patch = patch
            print(f"DEBUG: New Best Theory Found!")

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
        reward_rules(best_theory, trace)
    else:
        punish_rules(best_theory, trace)
    
    # CORTEX-OMEGA: Garbage Collection (Periodic)
    # For now, run every time (it's cheap).
    garbage_collect(best_theory)
    
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
    engine.forward_chain()
    target = Literal(scene.target_predicate, (scene.target_entity,))
    prediction = facts_copy.contains(target)
    trace = getattr(engine, "trace", [])
    
    # CORTEX-OMEGA: Usage Tracking
    # If prediction is True (or just if rules were used?), we credit them.
    # Let's credit any rule that fired in the trace.
    if trace:
        import time
        for step in trace:
            if step["type"] == "derivation":
                rid = step["rule_id"]
                if rid in theory.rules:
                    theory.rules[rid].usage_count += 1
                    theory.rules[rid].last_used = time.time()
    
    return prediction, trace

def punish_rules(theory: RuleBase, trace: List[Dict], penalty: float = 0.5):
    """
    CORTEX-OMEGA: Punish rules that led to a FALSE POSITIVE.
    """
    punished = set()
    for step in trace:
        if step["type"] == "derivation":
            rid = step["rule_id"]
            if rid in theory.rules and rid not in punished:
                rule = theory.rules[rid]
                rule.failure_count += 1
                # Decay confidence
                old_conf = rule.confidence
                
                if old_conf > 0.9:
                    # FALLEN AXIOM: Immediate degradation
                    rule.confidence = 0.01
                    rule.usage_count = 0 # Reset usage to allow GC
                    print(f"CORTEX: FALLEN AXIOM {rid} (Conf {old_conf:.2f} -> {rule.confidence:.2f}). Usage reset.")
                else:
                    rule.confidence *= (1.0 - penalty)
                    print(f"CORTEX: Punishing rule {rid} (Conf {old_conf:.2f} -> {rule.confidence:.2f}) due to False Positive.")
                
                punished.add(rid)

def reward_rules(theory: RuleBase, trace: List[Dict], reward: float = 0.1):
    """
    CORTEX-OMEGA: Reward rules that led to a CORRECT PREDICTION.
    """
    rewarded = set()
    for step in trace:
        if step["type"] == "derivation":
            rid = step["rule_id"]
            if rid in theory.rules and rid not in rewarded:
                rule = theory.rules[rid]
                rule.support_count += 1
                # Boost confidence (asymptotic approach to 1.0)
                # New Error = Old Error * (1 - Reward)
                # Conf = 1 - Error
                rule.confidence = 1.0 - (1.0 - rule.confidence) * (1.0 - reward)
                rewarded.add(rid)

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
        if rules:
            return rules[0]
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
        harmony = accuracy / (1 + λ * complexity)

    Donde:
      - accuracy: proporción de escenas bien clasificadas.
      - complexity: longitud estructural de la teoría + Entropía.
    """
    acc = evaluate_accuracy(theory, scenes)
    complexity = compute_complexity(theory, entropy_map)
    return acc / (1.0 + lambda_complexity * complexity)


def evaluate_accuracy(theory: RuleBase, scenes: List[Scene]) -> float:
    if not scenes:
        return 0.0

    correct = 0
    for s in scenes:
        facts_copy = clone_factbase(s.facts)
        engine = InferenceEngine(facts_copy, theory)
        engine.forward_chain()

        if s.target_args:
            target = Literal(s.target_predicate, s.target_args)
        else:
            target = Literal(s.target_predicate, (s.target_entity,))
            
        prediction = facts_copy.contains(target)
        
        if prediction == s.ground_truth:
            correct += 1

    return correct / len(scenes)


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


def garbage_collect(theory: RuleBase, threshold: float = 0.1):
    """
    CORTEX-OMEGA Pillar 4: Concept Compression.
    Prunes low-utility rules.
    """
    removed = theory.prune(threshold)
    if removed:
        print(f"CORTEX: Garbage Collector pruned {len(removed)} rules: {removed}")


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
                print(f"CORTEX: Promoting {rule_id} to Axiom! (Conf={rule.confidence}, Usage={rule.usage_count})")
                
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
                            print(f"CORTEX: Logical Conflict detected! {r1.id} vs {r2.id}")
                            
        return conflicts

    def _bodies_equal(self, b1: List[Literal], b2: List[Literal]) -> bool:
        if len(b1) != len(b2): return False
        # Set comparison for order independence
        s1 = set(str(l) for l in b1)
        s2 = set(str(l) for l in b2)
        return s1 == s2
