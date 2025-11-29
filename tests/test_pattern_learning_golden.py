"""
Test: Does Cortex-Omega really learn and recognize patterns?

We create a synthetic world:

  - Objects have:
      material ∈ {iron, lead, stone, balsa, plastic, wood, foam}
      is_heavy ∈ {True, False}
      plus noisy features: color, shape, random_id

  - True rule of the world:
      is_sink = (is_heavy == True) and (material != "balsa")

    i.e. heavy things sink, EXCEPT heavy balsa (it floats).

We then:

  1. Generate 200 random objects.
  2. Use 70% as training data, 30% as test data.
  3. Train Cortex-Omega with absorb_memory(..., target_label="sink").
  4. Query Cortex on the test objects with (material, is_heavy) only.
  5. Measure accuracy on unseen data.
  6. Show a few explanations to see what patterns it actually uses.
"""

import random
from cortex_omega import Cortex

# ---------------------------------------------------------------------
# 1) Synthetic data generator
# ---------------------------------------------------------------------

MATERIALS = ["iron", "lead", "stone", "balsa", "plastic", "wood", "foam"]
COLORS = ["red", "blue", "green", "yellow"]
SHAPES = ["sphere", "cube", "block", "sheet"]


def generate_object(idx: int):
    """
    Generate a single synthetic object with:
      - material
      - is_heavy  (derived from material, mostly)
      - is_sink   (true world rule: heavy AND not balsa)
      - noise features (color, shape, random_id)
    """
    material = random.choice(MATERIALS)

    # Basic "physical" intuition:
    if material in ["iron", "lead", "stone"]:
        is_heavy = True
    elif material == "balsa":
        # Balsa is light *in reality*, but we FORCE it to be "heavy"
        # so that it's a genuine exception: heavy but does NOT sink.
        is_heavy = True
    else:
        # plastic, wood, foam: randomly heavy or not
        is_heavy = random.random() < 0.3

    # True label: heavy things sink, except balsa
    is_sink = is_heavy and (material != "balsa")

    # Add pure noise features (Cortex *should* ignore these if it's sane)
    color = random.choice(COLORS)
    shape = random.choice(SHAPES)
    random_id = random.randint(1, 10_000)

    return {
        "id": f"obj_{idx}",
        "material": material,
        "is_heavy": is_heavy,
        "is_sink": is_sink,
        "color": color,
        "shape": shape,
        "random_id": random_id,
    }


def generate_dataset(n=200):
    return [generate_object(i) for i in range(n)]


# ---------------------------------------------------------------------
# 2) Train/Test split
# ---------------------------------------------------------------------

def train_test_split(data, train_ratio=0.7):
    random.shuffle(data)
    n_train = int(len(data) * train_ratio)
    return data[:n_train], data[n_train:]


# ---------------------------------------------------------------------
# 3) Run the experiment
# ---------------------------------------------------------------------

def run_experiment():
    print("=== Cortex-Omega Pattern Learning Test ===\n")
    random.seed(42)

    # Generate synthetic world
    data = generate_dataset(n=200)
    train, test = train_test_split(data, train_ratio=0.7)

    print(f"Total objects: {len(data)} (train={len(train)}, test={len(test)})")

    # Initialize Cortex
    brain = Cortex()

    # Train on the 'sink' concept
    print("\n[TRAIN] Absorbing training data for target 'sink'...")
    brain.absorb_memory(train, target_label="sink")

    # -----------------------------------------------------------------
    # Evaluate on test set
    # -----------------------------------------------------------------
    print("\n[TEST] Evaluating on unseen objects...")

    correct = 0
    total = len(test)

    # For confusion/inspection
    tp = tn = fp = fn = 0

    # Store a few interesting cases to show explanations
    sample_cases = []

    for obj in test:
        true_label = obj["is_sink"]

        # IMPORTANT:
        # We query ONLY with the relevant causal features (material, is_heavy)
        # Noise features (color, shape, random_id) are ignored at query time.
        result = brain.query(
            material=obj["material"],
            is_heavy=obj["is_heavy"],
            target="sink"
        )

        pred_label = bool(result.prediction)

        if pred_label == true_label:
            correct += 1

        if true_label and pred_label:
            tp += 1
        elif not true_label and not pred_label:
            tn += 1
        elif not true_label and pred_label:
            fp += 1
        elif true_label and not pred_label:
            fn += 1

        # Keep a few diverse examples to inspect later
        if len(sample_cases) < 10:
            sample_cases.append((obj, result))

    # Accuracy metrics
    accuracy = correct / total if total > 0 else 0.0

    print(f"\n[RESULTS]")
    print(f"Accuracy on unseen data: {accuracy*100:.2f}%")
    print(f"TP={tp}, TN={tn}, FP={fp}, FN={fn}")

    # -----------------------------------------------------------------
    # Show sample explanations
    # -----------------------------------------------------------------
    print("\n[EXPLANATIONS FOR SAMPLE TEST OBJECTS]")
    for obj, res in sample_cases:
        print("\n----------------------------------------")
        print(f"Object: id={obj['id']}")
        print(f"  material = {obj['material']}")
        print(f"  is_heavy = {obj['is_heavy']}")
        print(f"  TRUE is_sink = {obj['is_sink']}")
        print(f"  Prediction = {res.prediction} (conf={res.confidence:.2f})")
        print(f"  Explanation: {res.explanation}")
        # Proof may be very short or None, but we show it if present
        if res.proof is not None:
            print(f"  Proof: {res.proof}")
        else:
            print("  Proof: (no supporting rule; defaulted to negative)")

    print("\n=== Done. ===")

    # -----------------------------------------------------------------
    # B. Hard Assertions (Stability Checklist)
    # -----------------------------------------------------------------
    print("\n[STABILITY CHECKLIST]")
    
    # 1. Accuracy Threshold
    assert accuracy >= 0.9, f"Accuracy {accuracy} is below 0.9 threshold!"
    print("✅ Accuracy >= 0.9")

    # 2. Ensure we have heavy balsa in our samples to verify exception logic
    # (We might need to scan 'test' if 'sample_cases' missed them, but let's check samples first)
    # Actually, let's scan the whole test set for balsa correctness to be sure.
    
    balsa_test_cases = [obj for obj in test if obj["material"] == "balsa" and obj["is_heavy"]]
    if not balsa_test_cases:
        print("⚠️ No heavy balsa in test set! Skipping balsa assertion (bad luck in split).")
    else:
        # Check how many were predicted correctly (False)
        balsa_correct = 0
        for obj in balsa_test_cases:
            res = brain.query(material=obj["material"], is_heavy=obj["is_heavy"], target="sink")
            if res.prediction is False:
                balsa_correct += 1
        
        print(f"Heavy Balsa Stats: {balsa_correct}/{len(balsa_test_cases)} correct.")
        assert balsa_correct >= 1, "Engine failed to treat heavy balsa as exception (0 correct)."
        print("✅ Heavy Balsa Exception Verified")

    # -----------------------------------------------------------------
    # C. Rule Inspection
    # -----------------------------------------------------------------
    print("\n[THEORY INSPECTION]")
    rules = brain.inspect_rules(target="sink")
    for r in rules:
        print(r)
        
    # Sanity check: at least one rule mentions is_heavy
    has_heavy = any("is_heavy" in str(r) for r in rules)
    assert has_heavy, "Theory failed to learn 'is_heavy' dependency!"
    print("✅ Theory mentions 'is_heavy'")


if __name__ == "__main__":
    run_experiment()
