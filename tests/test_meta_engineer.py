import sys
import os
from cortex_omega import Cortex

def test_meta_engineer():
    print("[TEST] Initializing Cortex for Meta-Engineer Protocol...")
    brain = Cortex()

    # 1. Define a Meta-Rule
    # We want to flag high-value transactions from unverified sources.
    # Data shape: {"amount": 60000, "status": "unverified"}
    # Facts: amount(obj, 60000), status(obj, 'unverified')
    # Rule: review_required(X) :- amount(X, V), V > 50000, status(X, 'unverified')
    
    rule_str = "review_required(X) :- amount(X, V), V > 50000, status(X, unverified)"
    print(f"[TEST] Adding Meta-Rule: {rule_str}")
    brain.add_rule(rule_str)

    # 2. Ingest Data via Memory (Programmatic)
    # This simulates the system "remembering" or "absorbing" a specific event.
    print("[TEST] Absorbing memory...")
    data = [
        {"amount": 60000, "status": "unverified", "id": "tx_001"}, # Should trigger
        {"amount": 10000, "status": "verified", "id": "tx_002"}    # Should NOT trigger
    ]
    # We use a dummy target_label because absorb_memory expects one, 
    # but we are testing the rule we just added, not learning new ones.
    brain.absorb_memory(data, target_label="is_fraud")

    # 3. Verify Logic via Query
    print("\n[TEST] Querying Logic...")
    
    # Case 1: High Value + Unverified
    print("   Query: Amount=60000, Status=unverified")
    # We pass the facts explicitly to query to simulate a new observation matching the rule
    result_1 = brain.query(amount=60000, status="unverified", target="review_required")
    print(f"   Prediction: {result_1.prediction} (Confidence: {result_1.confidence:.2f})")
    print(f"   Reasoning:  {result_1.axiom}")

    # Case 2: Low Value + Verified
    print("\n   Query: Amount=10000, Status=verified")
    result_2 = brain.query(amount=10000, status="verified", target="review_required")
    print(f"   Prediction: {result_2.prediction}")

    # Assertions
    if result_1.prediction and not result_2.prediction:
        print("\n[SUCCESS] Meta-Engineer Protocol verified. Cortex correctly applied the manual rule.")
    else:
        print(f"\n[FAILURE] Logic mismatch. Expected True/False, got {result_1.prediction}/{result_2.prediction}")
        sys.exit(1)

if __name__ == "__main__":
    test_meta_engineer()
