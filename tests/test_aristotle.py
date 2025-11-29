from cortex_omega.api.client import Cortex
import sys

print(f"--- 🏛️  Running The Aristotle Test ---")

try:
    # 1. Initialize
    engine = Cortex()

    # 2. Inject Rule (The 'Brain' Test)
    # "X is mortal IF X is a man"
    print("Step 1: Teaching Logic (All men are mortal)...")
    engine.add_rule("mortal(X) :- man(X)")

    # 3. Inject Data (The 'Memory' Test)
    # "Socrates is a man"
    print("Step 2: Injecting Data (Socrates is a man)...")
    socrates_data = [{"id": "socrates", "is_man": True}]
    # We map the JSON key 'is_man' to the predicate 'man'
    engine.absorb_memory(socrates_data, target_label="man")

    # 4. Query (The 'Inference' Test)
    print("Step 3: Asking Question (Is Socrates mortal?)...")
    result = engine.query(id="socrates", target="mortal")

    if result.prediction is True:
        print(f"\n✅ PASSED: Socrates is confirmed mortal. (Confidence: {result.confidence})")
        print("    The logic engine is fully operational.")
    else:
        print(f"\n❌ FAILED: The engine thinks Socrates is immortal.")
        sys.exit(1)

except Exception as e:
    print(f"\n❌ CRASH: {e}")
    sys.exit(1)
