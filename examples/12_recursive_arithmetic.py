# 12_recursive_arithmetic.py
"""
PROJECT GÖDEL MAX v2 – Recursive Arithmetic
===========================================

Demonstrates learning arithmetic using structured terms (Peano axioms)
instead of flat strings.

Representation:
0 -> zero
1 -> s(zero)
2 -> s(s(zero))
...
"""

import sys
import os
sys.path.insert(0, os.getcwd())

from cortex_omega import Cortex
from cortex_omega.core.rules import Term, parse_term

def to_peano(n: int) -> str:
    if n == 0: return "zero"
    return f"s({to_peano(n-1)})"

def run_peano_arithmetic():
    print("=== PROJECT GÖDEL MAX v2: Recursive Arithmetic ===")
    
    brain = Cortex(mode="strict")
    
    # 1. Teach Addition
    # We want it to learn:
    # add(X, zero, X)
    # add(X, s(Y), s(Z)) :- add(X, Y, Z)
    
    from cortex_omega.core.rules import Scene, FactBase, Literal
    
    memory = []
    
    # Generate examples for 0..2 (Smaller dataset for speed)
    for x in range(3):
        for y in range(3):
            z = x + y
            if z > 4: continue # Keep universe small
            
            x_term = to_peano(x)
            y_term = to_peano(y)
            z_term = to_peano(z)
            
            # Positive example
            fb = FactBase()
            
            s_id = f"add_{x}_{y}"
            scene = Scene(
                id=s_id,
                facts=fb,
                target_entity="dummy", # Not used for relational target
                target_predicate="add",
                ground_truth=True,
                target_args=(parse_term(x_term), parse_term(y_term), parse_term(z_term))
            )
            memory.append(scene)
            
            # Negative examples (mutations)
            if z > 0:
                s_id_neg = f"add_{x}_{y}_neg1"
                scene_neg = Scene(
                    id=s_id_neg,
                    facts=FactBase(),
                    target_entity="dummy",
                    target_predicate="add",
                    ground_truth=False,
                    target_args=(parse_term(x_term), parse_term(y_term), parse_term(to_peano(z-1)))
                )
                memory.append(scene_neg)
                
            s_id_neg2 = f"add_{x}_{y}_neg2"
            scene_neg2 = Scene(
                id=s_id_neg2,
                facts=FactBase(),
                target_entity="dummy",
                target_predicate="add",
                ground_truth=False,
                target_args=(parse_term(x_term), parse_term(y_term), parse_term(to_peano(z+1)))
            )
            memory.append(scene_neg2)

    print(f"📚 Feeding {len(memory)} examples...")
    
    # Manually learn
    from cortex_omega.core.learner import Learner
    from cortex_omega.core.config import KernelConfig
    
    # Use the brain's config but ensure we have a learner
    learner = brain.learner
    
    # Limit iterations to prevent infinite recursion on "add" during learning (Critic evaluation)
    brain.config.inference_max_iterations = 2
    
    for i, scene in enumerate(memory):
        brain.theory, brain.memory = learner.learn(
            brain.theory, scene, brain.memory, brain.axioms
        )
    
    print("\n📜 Learned Rules:")
    rules = brain.inspect_rules("add")
    for r in rules:
        print(f"  - {r}")
        
    # 2. Query
    print("\n🧮 Testing Generalization (4+1=5)...")
    # Note: 5 was never seen in training (max z=4)
    # If it learned the recursive rule, it should handle this!
    
    # Construct query scene
    q_scene = Scene(
        id="query",
        facts=FactBase(),
        target_entity="dummy",
        target_predicate="add",
        ground_truth=False,
        target_args=(parse_term(to_peano(4)), parse_term(to_peano(1)), parse_term(to_peano(5)))
    )
    
    from cortex_omega.core.inference import infer
    
    # Limit iterations to prevent infinite recursion on "add"
    brain.config.inference_max_iterations = 2
    pred, trace = infer(brain.theory, q_scene, brain.config)
    
    print(f"  4 + 1 = 5? -> {pred}")
    if trace:
        print("  Trace:")
        for step in trace:
            print(f"    {step}")

if __name__ == "__main__":
    run_peano_arithmetic()
