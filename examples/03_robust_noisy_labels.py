"""
Example 03: Robustness to Noisy Labels
======================================
Demonstrates Cortex-Omega's ability to ignore spurious correlations (noise)
and focus on robust, causal features.

Scenario:
- Signal: "Square" objects are targets.
- Noise: "Red" objects are *mostly* targets (80% correlation), but not always.
- Cortex should learn "Square" and ignore "Red".
"""

from cortex_omega.api.client import Cortex

def main():
    # 1. Initialize
    cortex = Cortex(mode="robust")
    print("🧠 Cortex Initialized (Robust Mode).")

    # 2. Teach
    # We generate data where:
    # - Squares are ALWAYS targets.
    # - Red things are OFTEN targets (confounder).
    
    training_data = [
        # True Positives (Square + Red) - The trap!
        {"id": "obj1", "shape": "square", "color": "red", "target": True},
        {"id": "obj2", "shape": "square", "color": "red", "target": True},
        {"id": "obj3", "shape": "square", "color": "red", "target": True},
        
        # Crucial Counter-Example for Color (Red but NOT Square -> NOT Target)
        {"id": "obj4", "shape": "circle", "color": "red", "target": False},
        
        # Crucial Positive for Shape (Square but NOT Red -> Target)
        {"id": "obj5", "shape": "square", "color": "blue", "target": True},
        
        # Negatives
        {"id": "obj6", "shape": "circle", "color": "blue", "target": False},
    ]
    
    print(f"📚 Absorbing {len(training_data)} noisy examples...")
    cortex.absorb_memory(training_data, target_label="target")
    
    # 3. Query
    # We test with a "Blue Square" (Should be True) and a "Red Circle" (Should be False)
    
    test_objects = [
        {"id": "test_blue_square", "shape": "square", "color": "blue"}, # Expect True
        {"id": "test_red_circle", "shape": "circle", "color": "red"},   # Expect False
    ]
    
    print("\n🔮 Predictions:")
    for obj in test_objects:
        result = cortex.query(target="target", **obj)
        print(f"  Object {obj['id']} ({obj['shape']}, {obj['color']}): Target? {result.prediction}")
        if result.explanation:
            print(f"    Reason: {result.explanation}")

    # 4. Inspect Rules
    # We expect to see a rule about 'shape=square' and NO rule about 'color=red'.
    print("\n📜 Learned Rules:")
    for rule in cortex.export_rules():
        print(f"  {rule}")

if __name__ == "__main__":
    main()
