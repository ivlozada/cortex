import random
import sys
from cortex_omega import Cortex

# --- ANSI Colors for Terminal Output ---
GREEN = "\033[92m"
RED = "\033[91m"
MAGENTA = "\033[95m"
RESET = "\033[0m"

def print_header(msg): print(f"\n{MAGENTA}=== {msg} ==={RESET}")
def print_pass(msg): print(f"{GREEN}[PASS]{RESET} {msg}")
def print_fail(msg): print(f"{RED}[FAIL]{RESET} {msg}")

def run_legal_exception_test():
    print_header("CORTEX-Ω: Legal Exception Handling Protocol")
    
    # Initialize Cortex with low complexity penalty to allow specific exceptions
    brain = Cortex(sensitivity=0.1)
    
    # --- PHASE 1: The General Rule (Goliath) ---
    # Scenario: "Heavy objects SINK" (General Law)
    print_header("PHASE 1: Establishing General Law (Goliath)")
    print("Feeding 100 cases of 'Heavy Objects SINK'...")
    
    goliath_data = []
    materials = ["iron", "stone", "lead", "gold", "concrete"]
    
    for i in range(100):
        mat = random.choice(materials)
        # Data: Heavy, Material=X -> Sinks=True
        goliath_data.append({
            "mass": "heavy", 
            "material": mat, 
            "result": True   # Target: sinks
        })
        
    brain.absorb_memory(goliath_data, target_label="sinks")
    
    # Verify Goliath
    q_iron = brain.query(mass="heavy", material="iron", target="sinks")
    if q_iron.prediction:
        print_pass(f"General Law Established: Heavy objects sink. (Conf: {q_iron.confidence:.2f})")
    else:
        print_fail("Failed to learn General Law.")
        return

    # --- PHASE 2: The Exception (David) ---
    # Scenario: "Balsa Wood is HEAVY (in this context) but FLOATS" (Specific Exception)
    print_header("PHASE 2: Introducing Legal Exception (David)")
    print("Feeding 10 cases of 'Balsa Wood' (Heavy but Floats)...")
    
    david_data = []
    for i in range(10):
        # Data: Heavy, Material=Balsa -> Sinks=False
        david_data.append({
            "mass": "heavy", 
            "material": "balsa", 
            "result": False   # Target: sinks
        })
        
    brain.absorb_memory(david_data, target_label="sinks")
    print("Exception data ingested.")

    # --- PHASE 3: The Logic Duel ---
    print_header("PHASE 3: Inference Stress Test")
    
    # Test 1: Does the General Law still hold?
    print("1. Querying: Iron (Heavy)")
    q_goliath = brain.query(mass="heavy", material="iron", target="sinks")
    print(f"   Prediction: {q_goliath.prediction} (Expected: True)")
    
    # Test 2: Is the Exception respected?
    print("2. Querying: Balsa (Heavy)")
    q_david = brain.query(mass="heavy", material="balsa", target="sinks")
    print(f"   Prediction: {q_david.prediction} (Expected: False)")
    
    # Verdict
    if q_goliath.prediction is True and q_david.prediction is False:
        print_header("VERDICT: SUCCESS 🏆")
        print("The system successfully prioritized Specificity (Exception) over Frequency (General Rule).")
        print("This proves Cortex can handle complex legal frameworks with exceptions.")
    else:
        print_header("VERDICT: FAIL ❌")
        if q_david.prediction is True:
            print("System ignored the exception (Bully Logic).")
        if q_goliath.prediction is False:
            print("System forgot the general rule (Catastrophic Forgetting).")

if __name__ == "__main__":
    run_legal_exception_test()
