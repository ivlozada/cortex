from cortex_omega import Cortex

brain = Cortex()

try:
    # Test adding a rule with infix operator
    brain.add_rule("fraud(X) :- transaction(X, amount, V), V > 10000, unverified(X)")
    
    # Verify it's in the theory
    print("\nCurrent Theory:")
    print(brain.theory)
    
    print("\n[SUCCESS] Rule added successfully.")
except Exception as e:
    print(f"\n[FAILURE] {e}")
