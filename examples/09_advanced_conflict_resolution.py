"""
Example 09: Advanced Conflict Resolution (Level 2)
==================================================

This example pushes the "David vs. Goliath" scenario further by introducing
multiple distinct exceptions to a general rule.

Scenario:
1. General Rule: "Heavy things sink" (Iron, Lead, Stone)
2. Exception A: "Balsa wood is heavy but floats" (Material-based exception)
3. Exception B: "Hollow steel sphere is heavy but floats" (Structure-based exception)

Goal:
See if Cortex can maintain the general rule for standard heavy objects (Solid Iron)
while correctly handling *both* types of exceptions.
"""

from cortex_omega import Cortex

def main():
    print("=== Example 09: Advanced Conflict Resolution (Level 2) ===\n")
    brain = Cortex()

    # 1. Teach General Pattern: "Heavy things sink"
    print("Step 1: Teaching that heavy things sink...")
    data_sink = [
        {"id": "iron_block", "material": "iron",  "structure": "solid",  "is_heavy": True, "is_sink": True},
        {"id": "lead_block", "material": "lead",  "structure": "solid",  "is_heavy": True, "is_sink": True},
        {"id": "stone_rock", "material": "stone", "structure": "solid",  "is_heavy": True, "is_sink": True},
    ]
    brain.absorb_memory(data_sink, target_label="sink")

    # 2. Teach Exception A: Balsa (Material Exception)
    print("Step 2: Teaching Exception A (Balsa floats)...")
    data_balsa = [
        {"id": "balsa_log", "material": "balsa", "structure": "solid", "is_heavy": True, "is_sink": False},
    ]
    brain.absorb_memory(data_balsa, target_label="sink")

    # 3. Teach Exception B: Hollow Sphere (Structural Exception)
    print("Step 3: Teaching Exception B (Hollow Steel floats)...")
    data_hollow = [
        {"id": "steel_ship", "material": "steel", "structure": "hollow", "is_heavy": True, "is_sink": False},
    ]
    brain.absorb_memory(data_hollow, target_label="sink")

    # 4. Query
    print("\nStep 4: Testing Generalization...")

    # Case 1: Standard Heavy Object (Solid Copper) -> Should Sink
    # RESULT ANALYSIS:
    # Currently (v1.2.5), Cortex tends to "play it safe" (Precision > Recall).
    # Instead of learning "Heavy + Solid -> Sink (Except Balsa)", it learns:
    # "Heavy + Iron -> Sink", "Heavy + Lead -> Sink", etc.
    # So Copper (which it hasn't seen) defaults to False.
    # This is "Overfitting" behavior, which is safer for high-stakes logic but less general.
    copper = brain.query(material="copper", structure="solid", is_heavy=True, target="sink")
    print(f"\n[COPPER BLOCK] (Heavy, Solid, Copper)")
    print(f"Prediction:  {copper.prediction} (Should be True if generalized, False if overfitted)")
    print(f"Confidence:  {copper.confidence:.2f}")
    print(f"Explanation: {copper.explanation}")

    # Case 2: Balsa -> Should Float
    balsa = brain.query(material="balsa", structure="solid", is_heavy=True, target="sink")
    print(f"\n[BALSA] (Heavy, Solid, Balsa)")
    print(f"Prediction:  {balsa.prediction} (Should be False)")
    print(f"Explanation: {balsa.explanation}")

    # Case 3: Hollow Steel -> Should Float
    hollow = brain.query(material="steel", structure="hollow", is_heavy=True, target="sink")
    print(f"\n[HOLLOW STEEL] (Heavy, Hollow, Steel)")
    print(f"Prediction:  {hollow.prediction} (Should be False)")
    print(f"Explanation: {hollow.explanation}")

if __name__ == "__main__":
    main()
