"""
Example 02: Relational Learning (Grandparent)
=============================================
Demonstrates Cortex-Omega's ability to learn multi-hop relational rules.
Teaches the concept of "grandparent" from "parent" relations.

Scenario:
- A is parent of B.
- B is parent of C.
- Therefore, A is grandparent of C.
"""

from cortex_omega.api.client import Cortex
from cortex_omega.core.rules import Scene, FactBase
from cortex_omega.core.errors import EpistemicVoidError

def main():
    # 1. Initialize
    cortex = Cortex()
    print("🧠 Cortex Initialized.")

    # 2. Teach
    # We define a family tree.
    # Abe -> Homer -> Bart
    # Abe -> Homer -> Lisa
    # Mona -> Homer -> Bart
    
    # We need to manually construct scenes to define the relational target args (X, Z).
    # Cortex.absorb_memory() defaults to unary targets (Property Learning).
    # For Relational Learning, we use the lower-level Scene API exposed via the client.
    
    memory = []
    
    # Scene 1: Abe is grandparent of Bart
    fb1 = FactBase()
    fb1.add("parent", ("abe", "homer"))
    fb1.add("parent", ("homer", "bart"))
    s1 = Scene(
        id="s1", 
        facts=fb1, 
        target_entity="abe", 
        target_predicate="grandparent", 
        ground_truth=True, 
        target_args=("abe", "bart") # X=abe, Z=bart
    )
    memory.append(s1)
    
    # Scene 2: Homer is NOT grandparent of Bart (he is parent)
    fb2 = FactBase()
    fb2.add("parent", ("homer", "bart"))
    s2 = Scene(
        id="s2", 
        facts=fb2, 
        target_entity="homer", 
        target_predicate="grandparent", 
        ground_truth=False, 
        target_args=("homer", "bart")
    )
    memory.append(s2)
    
    # Scene 3: Another positive example (Mona -> Homer -> Lisa)
    fb3 = FactBase()
    fb3.add("parent", ("mona", "homer"))
    fb3.add("parent", ("homer", "lisa"))
    s3 = Scene(
        id="s3", 
        facts=fb3, 
        target_entity="mona", 
        target_predicate="grandparent", 
        ground_truth=True, 
        target_args=("mona", "lisa")
    )
    memory.append(s3)

    print("📚 Absorbing family tree and examples...")
    
    # Manually feed the learner
    for scene in memory:
        cortex.theory, cortex.memory = cortex.learner.learn(
            cortex.theory, scene, cortex.memory, cortex.axioms
        )
    
    # 3. Query
    print("\n🔮 Predictions:")
    # Test: Grampa Smurf -> Papa Smurf -> Smurfette
    
    fb_test = FactBase()
    fb_test.add("parent", ("grampa_smurf", "papa_smurf"))
    fb_test.add("parent", ("papa_smurf", "smurfette"))
    
    q_scene = Scene(
        id="q1",
        facts=fb_test,
        target_entity="grampa_smurf",
        target_predicate="grandparent",
        ground_truth=True, # Expected
        target_args=("grampa_smurf", "smurfette")
    )
    
    from cortex_omega.core.inference import infer
    prediction, trace = infer(cortex.theory, q_scene)
    
    print(f"  Grampa Smurf is grandparent of Smurfette? {prediction}")
    if prediction:
        # Find the explanation in the trace
        for step in reversed(trace):
            if step["type"] == "derivation" and step["derived"].predicate == "grandparent":
                rule_id = step["rule_id"]
                rule = cortex.theory.rules.get(rule_id)
                print(f"    Reason: {rule}")
                break

    print("\n📜 Learned Rules:")
    for rule in cortex.export_rules(format="prolog").split("\n"):
        print(f"  {rule}")

if __name__ == "__main__":
    main()
