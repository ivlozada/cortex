"""
Example 01: Binary Classification
=================================
A simple "Hello World" for Cortex-Omega.
Teaches the system to classify objects as "hazardous" based on a single feature.

Scenario:
- Red objects are hazardous.
- Blue objects are safe.
"""

from cortex_omega.api.client import Cortex

def main():
    # 1. Initialize Cortex
    # We use a temporary workspace for this example.
    cortex = Cortex(workspace="./temp_workspace_01")
    
    print("🧠 Cortex Initialized.")

    # 2. Teach (Absorb Data)
    # We provide a few examples.
    training_data = [
        {"id": "obj1", "color": "red", "hazardous": True},
        {"id": "obj2", "color": "blue", "hazardous": False},
        {"id": "obj3", "color": "red", "hazardous": True},
        {"id": "obj4", "color": "green", "hazardous": False}, # Green is safe too
    ]
    
    print(f"📚 Absorbing {len(training_data)} training examples...")
    cortex.absorb(training_data, target_label="hazardous")
    
    # 3. Query (Inference)
    # Now we ask about new objects.
    test_objects = [
        {"id": "test1", "color": "red"},   # Should be True
        {"id": "test2", "color": "blue"},  # Should be False
        {"id": "test3", "color": "green"}, # Should be False
    ]
    
    print("\n🔮 Predictions:")
    for obj in test_objects:
        # We pass the object's features to query()
        # We ask: is it hazardous?
        result = cortex.query(target="hazardous", **obj)
        
        print(f"  Object {obj['id']} ({obj['color']}): Hazardous? {result.prediction}")
        if result.explanation:
            print(f"    Reason: {result.explanation}")

    # 4. Inspect Rules
    print("\n📜 Learned Rules:")
    for rule in cortex.export_rules():
        print(f"  {rule}")

if __name__ == "__main__":
    main()
