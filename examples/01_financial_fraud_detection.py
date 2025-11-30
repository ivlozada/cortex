import csv
import random
import os
import sys
from cortex_omega import Cortex

# --- STEP 1: Generate "Corporate Chaos" (Simulate messy real-world data) ---
def generate_dirty_csv(filename="financial_chaos.csv"):
    headers = ["TransactionID", "Amount_Raw", "Location_Mixed", "User_Type", "IS_FRAUD"]
    
    data = []
    print(f"[GENERATOR] Creating messy corporate data in '{filename}'...")
    
    for i in range(100):
        # 1. Noise Column (Location): Inconsistent formatting
        loc = random.choice(["NY", "New York", "ny", "N.Y.", "London", "LDN"])
        
        # 2. The Hidden Pattern (The Fraud Rule)
        # Rule: If Amount > 10000 AND User_Type = "Guest" -> FRAUD
        
        # Generate values
        user_type = random.choice(["Admin", "Guest", "User", "guest", "GUEST"]) # Inconsistent casing
        is_guest = user_type.lower() == "guest"
        
        # Dirty Amount (mix of strings and numbers)
        amount_val = random.randint(100, 20000)
        if amount_val > 10000:
            amount_str = random.choice([f"${amount_val}", f"{amount_val}", f"{amount_val}.00"])
            amount_category = "heavy" # Mapping internally to 'heavy' for CORTEX metaphor
        else:
            amount_str = f"{amount_val}"
            amount_category = "light"
            
        # Determine Ground Truth
        is_fraud = "YES" if (amount_category == "heavy" and is_guest) else "no"
        
        # Add random label noise (Human error)
        if random.random() < 0.02: 
            is_fraud = "YES" if is_fraud == "no" else "no"

        data.append([f"TX_{i:03d}", amount_str, loc, user_type, is_fraud])

    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(data)
    print("[GENERATOR] Done. 100 rows of chaotic data written.")

# --- EXECUTION ---
if __name__ == "__main__":
    # 1. Create the messy data file
    csv_file = "financial_chaos.csv"
    generate_dirty_csv(csv_file)
    
    # 2. Initialize Cortex
    print(f"\n[CORTEX] Initializing Neuro-Symbolic Kernel...")
    brain = Cortex()
    
    # 3. Absorb Data (The "Universal Ingestor")
    # Cortex automatically handles parsing, normalization, and noise filtering.
    brain.absorb(csv_file)
    
    # 4. Verify Logic
    print("\n[VERIFICATION] Querying the crystallized logic...")
    
    # Case A: Guest with Heavy Transaction (Should be Fraud)
    print("   Query: Guest + Heavy Transaction")
    try:
        result_fraud = brain.query(type="guest", mass="heavy", target="fraud")
        print(f"   Prediction: {result_fraud.prediction} (Confidence: {result_fraud.confidence:.2f})")
        print(f"   Reasoning:  {result_fraud.explanation}")
    except Exception as e:
        print(f"   [!] Query failed: {e}")
        result_fraud = None
    
    # Case B: Admin with Heavy Transaction (Should be Safe)
    print("\n   Query: Admin + Heavy Transaction")
    try:
        result_safe = brain.query(type="admin", mass="heavy", target="fraud")
        print(f"   Prediction: {result_safe.prediction}")
    except Exception as e:
        print(f"   [!] Query failed: {e}")
        result_safe = None
    
    if result_fraud and result_safe and result_fraud.prediction and not result_safe.prediction:
        print("\n[SUCCESS] Cortex successfully detected the hidden fraud pattern amidst the noise. 🚀")
    else:
        print("\n[FAIL] Logic extraction failed or queries were void.")
