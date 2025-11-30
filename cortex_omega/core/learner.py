from dataclasses import dataclass
from typing import List, Tuple, Any, Optional, Dict
import logging

from .rules import RuleBase, Scene
from .values import ValueBase
from .config import KernelConfig
from .theorist import Theorist
from .critic import Critic
from .hypothesis import FailureContext
from .engine import (
    infer,
    update_rule_stats,
    append_to_memory,
    garbage_collect,
    calculate_attribute_entropy,
    find_worst_error,
    violates_axioms
)

logger = logging.getLogger(__name__)

@dataclass
class Learner:
    config: KernelConfig

    def learn(
        self,
        theory: RuleBase,
        scene: Scene,
        memory: List[Scene],
        axioms: ValueBase
    ) -> Tuple[RuleBase, List[Scene]]:
        """
        El corazón del algoritmo.
        Entrada: teoría actual, nueva escena, memoria, axiomas y configuración.
        Salida: nueva teoría, nueva memoria.
        """
        logger.debug(f"DEBUG: Start Update Theory for Scene {scene.id}")

        # 1. Proyección
        prediction, trace = infer(theory, scene, self.config)

        # 2. Refuerzo / Castigo
        is_correct = (prediction == scene.ground_truth)
        update_rule_stats(theory, trace, is_correct, self.config)
        
        # Tipo de error
        if prediction and not scene.ground_truth:
            error_type = "FALSE_POSITIVE"
        elif not prediction and scene.ground_truth:
            error_type = "FALSE_NEGATIVE"
        elif not prediction and not scene.ground_truth:
            error_type = "NEGATIVE_GAP"
        else:
            # True Positive
            new_memory = append_to_memory(memory, scene, self.config.max_memory)
            return theory, new_memory

        # 2. Generación de teorías candidatas
        candidate_theories, ctx = self._propose_candidates(theory, scene, memory, trace, error_type)

        # 3. Evaluación y Selección
        eval_scenes = memory + [scene]
        entropy_map = calculate_attribute_entropy(eval_scenes)
        
        best_theory, best_patch = self._evaluate_and_select(
            theory, candidate_theories, eval_scenes, axioms, entropy_map, ctx
        )

        # 5. Actualizar memoria (FIFO)
        new_memory = append_to_memory(memory, scene, self.config.max_memory)
        
        # CORTEX-OMEGA: Reinforcement Learning (Update Confidence)
        pred, trace = infer(best_theory, scene, self.config)
        is_correct = (pred == scene.ground_truth)
        update_rule_stats(best_theory, trace, is_correct, self.config)
        
        # 5. Refinement Loop
        best_theory = self._refine_theory(best_theory, new_memory, eval_scenes, axioms, entropy_map)

        # CORTEX-OMEGA: Garbage Collection
        threshold = self.config.plasticity.get("min_conf_to_keep", 0.0) if self.config.plasticity else 0.0
        garbage_collect(best_theory, threshold=threshold, config=self.config)
        
        # CORTEX-OMEGA: Self-Reflecting Compiler
        if self.config.compiler:
            self.config.compiler.promote_stable_rules(best_theory, axioms)
            self.config.compiler.check_conflicts(best_theory)
        
        return best_theory, new_memory

    def _propose_candidates(
        self,
        theory: RuleBase,
        scene: Scene,
        memory: List[Scene],
        trace,
        error_type: str
    ) -> Tuple[List[Tuple[RuleBase, Any]], Optional[FailureContext]]:
        patch_generator = self.config.patch_generator
        if patch_generator is None:
            # Avoid circular import or instantiate default
            from .hypothesis import HypothesisGenerator
            patch_generator = HypothesisGenerator()
            
        theorist = Theorist(config=self.config, hypothesis_generator=patch_generator)
        return theorist.propose(theory, scene, memory, trace, error_type)

    def _evaluate_and_select(
        self,
        theory: RuleBase,
        candidate_theories: List[Tuple[RuleBase, Any]],
        eval_scenes: List[Scene],
        axioms: ValueBase,
        entropy_map: Dict[str, float],
        ctx: Optional[FailureContext]
    ) -> Tuple[RuleBase, Any]:
        critic = Critic(self.config)
        best_theory = theory
        best_harmony = critic.score_harmony(theory, eval_scenes, entropy_map)
        best_patch = None
        
        for i, (T_candidate, patch) in enumerate(candidate_theories):
            if violates_axioms(T_candidate, axioms, eval_scenes):
                continue

            h_new = critic.score_harmony(T_candidate, eval_scenes, entropy_map)

            accepted = False
            if h_new > best_harmony:
                accepted = True
            else:
                import math
                import random
                
                delta = h_new - best_harmony
                temp = max(self.config.temperature, self.config.sa_min_temp)
                prob = math.exp(delta / temp)
                
                if random.random() < prob:
                    if h_new < self.config.sa_acceptance_threshold and best_harmony > self.config.sa_acceptance_threshold:
                        pass
                    else:
                        logger.debug(f"Accepted worse candidate (Prob={prob:.4f}, T={temp:.4f}) to escape local max.")
                        accepted = True
            
            if accepted:
                best_harmony = h_new
                best_theory = T_candidate
                best_patch = patch

        self.config.temperature *= self.config.cooling_rate
                
        if best_patch and self.config.patch_generator and ctx:
            self.config.patch_generator.record_feedback(ctx, best_patch)
            
        return best_theory, best_patch

    def _refine_theory(
        self,
        theory: RuleBase,
        memory: List[Scene],
        eval_scenes: List[Scene],
        axioms: ValueBase,
        entropy_map: Dict[str, float]
    ) -> RuleBase:
        best_theory = theory
        critic = Critic(self.config)
        
        for i in range(self.config.max_refinement_steps):
            worst_scene = find_worst_error(best_theory, memory, axioms)
            
            if not worst_scene:
                break
                
            pred_r, trace_r = infer(best_theory, worst_scene, self.config)
            err_type = "FALSE_POSITIVE" if pred_r and not worst_scene.ground_truth else "FALSE_NEGATIVE"
            
            refinement_candidates, _ = self._propose_candidates(
                theory=best_theory,
                scene=worst_scene,
                memory=memory,
                trace=trace_r,
                error_type=err_type
            )
            
            if not refinement_candidates:
                break
                
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
