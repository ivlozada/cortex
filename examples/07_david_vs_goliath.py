"""
Example 07: David vs. Goliath (Conflict Resolution)
===================================================

This example demonstrates Cortex-Omega's ability to handle logical conflicts
where a specific exception overrides a general rule.

Scenario:
1. General Rule: "Heavy things sink" (Learned from Iron, Lead, Stone)
2. Exception: "Balsa wood is heavy but floats" (Learned from Balsa)

The Key Mechanism:
- Both Iron and Balsa are heavy.
- Cortex learns a general "heavy => sink" rule.
- Balsa, as a negative example for sink, triggers **contrastive refinement**.
- After refinement, the rule becomes specific (e.g., "heavy AND iron => sink"), 
  so a heavy Iron sinks, but a heavy Balsa does not.
"""

from cortex_omega import Cortex

def main():
    print("=== Example 07: David vs. Goliath (Conflict Resolution) ===\n")
    brain = Cortex()

    # 1. Teach General Pattern: "Heavy things sink"
    print("Step 1: Teaching that heavy things (Iron, Lead, Stone) sink...")
    data_sink = [
        {"id": "iron_ball",  "material": "iron",   "is_heavy": True,  "is_sink": True},
        {"id": "lead_block", "material": "lead",   "is_heavy": True,  "is_sink": True},
        {"id": "stone",      "material": "stone",  "is_heavy": True,  "is_sink": True},
        {"id": "ping_pong",  "material": "plastic","is_heavy": False, "is_sink": False},
    ]
    brain.absorb_memory(data_sink, target_label="sink")

    # 2. Teach Exception: "Balsa is heavy but does NOT sink"
    # Note: We must use is_sink=False to teach the negation of the target concept.
    print("Step 2: Teaching exception (Balsa is heavy but does NOT sink)...")
    data_exception = [
        {"id": "balsa_block", "material": "balsa", "is_heavy": True, "is_sink": False},
    ]
    brain.absorb_memory(data_exception, target_label="sink")

    # 3. Query
    # We must provide 'is_heavy=True' in the query because the engine learned that 'heavy' is the cause.
    print("\nStep 3: Testing conflict resolution...")
    
    # Case A: Iron (Should follow General Rule)
    iron = brain.query(material="iron", is_heavy=True, target="sink")
    print(f"\n[IRON] (Heavy=True, Material=Iron)")
    print(f"Prediction:  {iron.prediction} (Should be True)")
    print(f"Confidence:  {iron.confidence:.2f}")
    print(f"Explanation: {iron.explanation}")
    print(f"Proof:       {iron.proof}")

    # Case B: Balsa (Should follow Exception)
    balsa = brain.query(material="balsa", is_heavy=True, target="sink")
    print(f"\n[BALSA] (Heavy=True, Material=Balsa)")
    print(f"Prediction:  {balsa.prediction} (Should be False)")
    print(f"Confidence:  {balsa.confidence:.2f}")
    print(f"Explanation: {balsa.explanation}")
    print(f"Proof:       {balsa.proof}")

if __name__ == "__main__":
    main()
