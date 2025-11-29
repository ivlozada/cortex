
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cortex_omega import Cortex
from cortex_omega.core.rules import FactBase, Scene

def run_demo():
    print("="*60)
    print("CORTEX OMEGA v1.4: CORE CAPABILITIES DEMO")
    print("="*60)
    
    print("="*60)
    
    import logging
    logging.getLogger().setLevel(logging.DEBUG)
    logging.basicConfig(level=logging.DEBUG, force=True)
    
    brain = Cortex()
    
    # ==========================================
    # 1. Rule Learning (Categorical)
    # ==========================================
    print("\n[1] Demonstrating Rule Learning (Categorical)")
    print("    Goal: Learn that 'is_senior(X)' implies 'age_group(X, senior)'")
    
    # Train: Seniors
    train_data = []
    for i in range(5):
        train_data.append({
            "id": f"pos_{i}", 
            "age_group": "senior", 
            "is_senior": True
        })
        
    # Train: Juniors
    for i in range(5):
        train_data.append({
            "id": f"neg_{i}", 
            "age_group": "junior", 
            "is_senior": False
        })
        
    brain.absorb_memory(train_data, target_label="is_senior")
    
    # Test: New Senior
    test_case = {"id": "test_senior", "age_group": "senior"}
    prediction = brain.query(**test_case, target="is_senior")
    
    print(f"    Query(AgeGroup=Senior) -> Predicted Senior? {prediction.prediction} (Conf: {prediction.confidence:.2f})")
    
    # Inspect Rules
    rules = brain.inspect_rules(target="is_senior")
    print("    Learned Rules:")
    for r in rules:
        print(f"      - {r}")
        
    # ==========================================
    # 2. Confounder Invariance (v1.4)
    # ==========================================
    print("\n[2] Demonstrating Confounder Invariance (Stability Check)")
    print("    Goal: Distinguish Causal Feature (Shape) from Confounder (Color)")
    print("    Scenario: 'Color=Red' works early but fails later. 'Shape=Square' always works.")
    
    brain_conf = Cortex()
    
    # Phase 1: Early (Red & Square -> True)
    early_data = []
    for i in range(5):
        early_data.append({
            "id": f"early_{i}",
            "color": "red",
            "shape": "square",
            "glows": True
        })
    brain_conf.absorb_memory(early_data, target_label="glows")
    
    # Phase 2: Late (Red -> False, Square -> True)
    late_data = []
    # Red but Circle -> False (Red is no longer reliable)
    for i in range(3):
        late_data.append({
            "id": f"late_a_{i}",
            "color": "red",
            "shape": "circle",
            "glows": False
        })
    # Blue but Square -> True (Square is still reliable)
    for i in range(3):
        late_data.append({
            "id": f"late_b_{i}",
            "color": "blue",
            "shape": "square",
            "glows": True
        })
        
    brain_conf.absorb_memory(late_data, target_label="glows")
    
    # Inspect Rules
    print("    Learned Rules (Should prefer Shape over Color):")
    rules = brain_conf.inspect_rules(target="glows")
    for r in rules:
        print(f"      - {r} (Rel={r.reliability:.2f}, Score={r.fires_pos - r.fires_neg})")
        
    # ==========================================
    # 3. Transparent Auditing (v1.4)
    # ==========================================
    print("\n[3] Demonstrating Transparent Auditing (Rule Stats)")
    print("    Goal: Inspect internal statistics of the learned rules.")
    
    for r in rules:
        if "shape" in str(r):
            print(f"    Rule: {r}")
            print(f"      - Positive Firings: {r.fires_pos}")
            print(f"      - Negative Firings: {r.fires_neg}")
            print(f"      - Reliability:      {r.reliability:.2f}")
            print(f"      - Complexity:       {r.complexity}")
            print(f"      - MDL Score:        {(r.fires_pos - r.fires_neg) - (0.2 * r.complexity):.2f}")

    print("\n" + "="*60)
    print("DEMO COMPLETE")
    print("="*60)

if __name__ == "__main__":
    run_demo()
