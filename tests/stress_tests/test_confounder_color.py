"""
Test: Confounder Trap with Color

Train set:
  - We bias the data so that most sinking objects are "red".
  - This makes "color=red" *look* predictive.

Test set:
  - Color is random and not correlated with sink.
  - True world rule remains:
        is_sink = is_heavy and (material != "balsa")

If Cortex overfits the confounder:
  - Accuracy will drop on test.
  - Explanations may mention 'color'.

If Cortex learns the real pattern:
  - Accuracy stays high.
  - Final rule will mention is_heavy/material, not color.
"""

import random
from cortex_omega import Cortex

MATERIALS = ["iron", "lead", "stone", "balsa", "plastic", "wood", "foam"]
COLORS = ["red", "blue", "green", "yellow"]
SHAPES = ["sphere", "cube", "block", "sheet"]


def true_world_object(idx: int, biased: bool):
    """
    If biased=True -> training mode (color skewed).
    If biased=False -> test mode (color random).
    """
    material = random.choice(MATERIALS)

    # Same heaviness + true rule as before
    if material in ["iron", "lead", "stone"]:
        is_heavy = True
    elif material == "balsa":
        is_heavy = True  # forced to be heavy -> exception
    else:
        is_heavy = random.random() < 0.4

    is_sink = is_heavy and (material != "balsa")

    # Color:
    if biased:
        # If it sinks, 80% chance it's 'red'
        if is_sink and random.random() < 0.8:
            color = "red"
        else:
            color = random.choice(COLORS)
    else:
        # In test, color is completely random
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


def generate_train_test(n_train=150, n_test=80):
    random.seed(999)
    train = [true_world_object(i, biased=True) for i in range(n_train)]
    test = [true_world_object(10_000 + i, biased=False) for i in range(n_test)]
    return train, test


def run_experiment():
    print("=== Confounder (Color) Test ===\n")

    train, test = generate_train_test(n_train=150, n_test=80)
    print(f"Train={len(train)}, Test={len(test)}")

    brain = Cortex()
    print("\n[TRAIN] Absorbing biased training data (color is confounded)...")
    brain.absorb_memory(train, target_label="sink")

    print("\n=== Training Complete ===")
    print(brain.theory)
    
    print("\n[TEST] Evaluating on unbiased test data...")

    tp = tn = fp = fn = 0
    sample_cases = []
    any_color_rule = False

    for obj in test:
        true_label = obj["is_sink"]

        res = brain.query(
            material=obj["material"],
            is_heavy=obj["is_heavy"],
            target="sink",
        )
        pred_label = bool(res.prediction)

        if pred_label and true_label:
            tp += 1
        elif not pred_label and not true_label:
            tn += 1
        elif pred_label and not true_label:
            fp += 1
        elif not pred_label and true_label:
            fn += 1

        if "color" in (res.explanation or ""):
            any_color_rule = True

        if len(sample_cases) < 12:
            sample_cases.append((obj, res))

    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total else 0.0

    print(f"\n[RESULTS]")
    print(f"Accuracy on unbiased test: {acc*100:.2f}%  (TP={tp}, TN={tn}, FP={fp}, FN={fn})")
    print(f"Did any explanation mention 'color'? {'YES' if any_color_rule else 'NO'}")

    print("\n[EXPLANATIONS - SAMPLE TEST OBJECTS]")
    for obj, res in sample_cases:
        print("\n----------------------------------------")
        print(f"id={obj['id']}, material={obj['material']}, is_heavy={obj['is_heavy']}, color={obj['color']}")
        print(f"TRUE is_sink = {obj['is_sink']}")
        print(f"Prediction   = {res.prediction} (conf={res.confidence:.2f})")
        print(f"Explanation  = {res.explanation}")
        if res.proof is not None:
            print(f"Proof        = {res.proof}")
        else:
            print("Proof        = (no supporting rule; defaulted to negative)")

    print("\n=== Done. ===")


if __name__ == "__main__":
    run_experiment()
