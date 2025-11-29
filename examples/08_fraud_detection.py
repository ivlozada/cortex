"""
Cortex-Omega Example: Fraud Detection
=====================================
Demonstrates how to use Cortex for anomaly detection in financial transactions.
"""

import logging
from cortex_omega import Cortex

# Enable logging to see the engine in action (optional)
# logging.basicConfig(level=logging.INFO)

def main():
    print("🤖 Initializing Cortex Fraud Engine...")
    brain = Cortex()

    # 1. Teach the Rules (or let it learn from data)
    # Here we seed it with some expert knowledge
    print("📚 Injecting expert knowledge...")
    brain.add_rule("fraud(X) :- high_risk_country(X), amount_large(X)")
    brain.add_rule("fraud(X) :- velocity_high(X)")
    
    # Exception: Trusted users are rarely fraud
    # Note: Cortex learns exceptions from data, but we can also hint them.
    # For now, let's rely on data to learn the exception.

    # 2. Absorb Data (Training)
    print("🧠 Absorbing transaction history...")
    training_data = [
        # Normal patterns
        {"id": "tx1", "amount_large": False, "high_risk_country": False, "fraud": False},
        {"id": "tx2", "amount_large": True, "high_risk_country": False, "fraud": False},
        
        # Fraud patterns
        {"id": "tx3", "amount_large": True, "high_risk_country": True, "fraud": True},
        {"id": "tx4", "velocity_high": True, "fraud": True},
        
        # The Exception: VIP user in high risk country doing large tx might be OK?
        # Let's say VIP overrides high risk.
        {"id": "tx5", "amount_large": True, "high_risk_country": True, "is_vip": True, "fraud": False},
        {"id": "tx6", "amount_large": True, "high_risk_country": True, "is_vip": True, "fraud": False},
    ]
    
    brain.absorb_memory(training_data, target_label="fraud")

    # 3. Query (Inference)
    print("\n🔍 Running Inference...")
    
    # Case A: High Risk, Large Amount, Not VIP -> Should be Fraud
    case_a = {"id": "new_tx_1", "amount_large": True, "high_risk_country": True, "is_vip": False}
    result_a = brain.query(**case_a, target="fraud")
    print(f"Case A (Risk+Large): {result_a.prediction} (Conf: {result_a.confidence:.2f})")
    print(f"  Explanation: {result_a.explanation}")
    
    # Case B: High Risk, Large Amount, BUT VIP -> Should be Safe (Exception)
    case_b = {"id": "new_tx_2", "amount_large": True, "high_risk_country": True, "is_vip": True}
    result_b = brain.query(**case_b, target="fraud")
    print(f"Case B (Risk+Large+VIP): {result_b.prediction} (Conf: {result_b.confidence:.2f})")
    if not result_b.prediction:
        print(f"  Explanation: {result_b.explanation}")

    # Case C: Just Velocity -> Fraud
    case_c = {"id": "new_tx_3", "velocity_high": True}
    result_c = brain.query(**case_c, target="fraud")
    print(f"Case C (Velocity): {result_c.prediction} (Conf: {result_c.confidence:.2f})")
    print(f"  Explanation: {result_c.explanation}")

if __name__ == "__main__":
    main()
