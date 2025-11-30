import pytest
from cortex_omega.core.engine import infer, update_theory_kernel, KernelConfig
from cortex_omega.core.rules import RuleBase, Scene, FactBase, Literal, Rule

def test_binary_negation_inference():
    """
    Test that NOT_father(X, Y) respects arity 2.
    """
    # 1. Setup: Theory with a negative binary rule
    # NOT_father(X, Y) :- age_diff(X, Y, small)
    
    theory = RuleBase()
    rule = Rule(
        id="R_not_father_small_diff",
        head=Literal("NOT_father", ("X", "Y")),
        body=(Literal("age_diff", ("X", "Y", "small")),)
    )
    theory.add(rule)
    
    # 2. Scene: X and Y have small age difference
    facts = FactBase()
    facts.add("age_diff", ("alice", "bob", "small"))
    
    scene = Scene(
        id="scene_1",
        facts=facts,
        target_entity="alice", # Legacy field, might be ignored for binary
        target_predicate="father",
        target_args=("alice", "bob"),
        ground_truth=False
    )
    
    # 3. Infer
    # This should trigger the negative rule and return False (prediction)
    # If it constructs NOT_father("alice"), it won't match NOT_father("alice", "bob")
    
    pred, trace = infer(theory, scene)
    
    # Check if the rule fired
    fired = False
    for step in trace:
        if step["type"] == "derivation" and step["rule_id"] == "R_not_father_small_diff":
            fired = True
            break
            
    assert fired, "Negative binary rule should have fired"
    assert pred is False, "Prediction should be False due to negative rule"

def test_binary_negation_learning():
    """
    Test that the kernel can learn a negative binary rule.
    """
    config = KernelConfig()
    theory = RuleBase()
    memory = []
    from cortex_omega.core.values import ValueBase
    axioms = ValueBase()
    
    # Scene 1: Father(alice, bob) -> False (because age diff is small)
    facts = FactBase()
    facts.add("age_diff", ("alice", "bob", "small"))
    
    scene = Scene(
        id="scene_1",
        facts=facts,
        target_entity="alice",
        target_predicate="father",
        target_args=("alice", "bob"),
        ground_truth=False
    )
    
    # Update theory
    theory, memory = update_theory_kernel(theory, scene, memory, axioms, config)
    
    # Check if we learned something useful (or at least didn't crash)
    # Since it's a False Negative (we predicted False? No, empty theory predicts False by default)
    # Wait, empty theory -> infer returns False. GT is False.
    # So it's a True Negative (default).
    # We need a case where it predicts True incorrectly, or we want to learn the negative rule explicitly?
    # If GT is False and Pred is False, it might not learn anything unless we force it.
    
    # Let's try a False Positive case:
    # Theory says father(X,Y) :- male(X).
    # We have male(alice) (wait alice is female, say 'charlie').
    # But age diff is small, so NOT father.
    
    theory.add(Rule("R_father_male", Literal("father", ("X", "Y")), (Literal("male", ("X")),)))
    
    facts2 = FactBase()
    facts2.add("male", ("charlie",))
    facts2.add("age_diff", ("charlie", "dave", "small"))
    
    scene2 = Scene(
        id="scene_2",
        facts=facts2,
        target_entity="charlie",
        target_predicate="father",
        target_args=("charlie", "dave"),
        ground_truth=False # Contradicts R_father_male
    )
    
    # This should trigger a False Positive correction
    theory, memory = update_theory_kernel(theory, scene2, memory, axioms, config)
    
    # Verify that we learned a negative rule or exception
    # NOT_father(X, Y) :- age_diff(X, Y, small)
    
    # Just check that now it predicts False
    pred, _ = infer(theory, scene2)
    assert pred is False, "Should have corrected the False Positive for binary relation"
