from cortex_omega.api.client import Cortex
import sys

print("--- ⛓️  Running The Transitivity Chain Test ---")

try:
    # 1. Initialize
    engine = Cortex()

    # 2. Teach the Rules (The Logic Chain)
    print("Step 1: Teaching Logic...")
    # Rule A: All Greeks are Human
    print("   -> Rule: human(X) :- greek(X)")
    engine.add_rule("human(X) :- greek(X)")
    
    # Rule B: All Humans are Mortal
    print("   -> Rule: mortal(X) :- human(X)")
    engine.add_rule("mortal(X) :- human(X)")

    # 3. Inject Data
    print("Step 2: Injecting Data (Socrates is Greek)...")
    # We ONLY tell it he is Greek. We do NOT say he is Human or Mortal.
    engine.absorb_memory([{"id": "socrates", "is_greek": True}], target_label="greek")

    # 4. The Query (Requires jumping from Greek -> Human -> Mortal)
    print("Step 3: Asking Question (Is Socrates mortal?)...")
    result = engine.query(id="socrates", target="mortal")

    if result.prediction:
        print(f"\n✅ PASSED: Logic bridged the gap! (Confidence: {result.confidence})")
        # Check if explanation exists before printing
        if hasattr(result, 'explanation') and result.explanation:
             print(f"   Explanation: {result.explanation}")
    else:
        print(f"\n❌ FAILED: Could not connect the dots.")

except Exception as e:
    print(f"\n❌ CRASH: {e}")
