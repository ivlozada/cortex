"""
Test: Multi-Exception Pattern

True world rule:
    is_sink = is_heavy and material not in {"balsa", "foam"}

So:
  - Heavy iron/lead/stone/wood -> sink
  - Heavy balsa -> does NOT sink (exception #1)
  - Heavy foam -> does NOT sink (exception #2)
  - Non-heavy anything -> does NOT sink

We want Cortex to:
  - Learn a rule that uses is_heavy and excludes balsa + foam
  - Generalize correctly to unseen objects
"""

import random
from cortex_omega import Cortex


MATERIALS = ["iron", "lead", "stone", "balsa", "plastic", "wood", "foam"]
COLORS = ["red", "blue", "green", "yellow"]
SHAPES = ["sphere", "cube", "block", "sheet"]


def generate_object(idx: int):
    material = random.choice(MATERIALS)

    # Define heaviness
    if material in ["iron", "lead", "stone"]:
        is_heavy = True
    elif material in ["balsa", "foam"]:
        # Force them to be heavy so they're genuine exceptions
        is_heavy = True
    else:
        # plastic, wood: sometimes heavy
        is_heavy = random.random() < 0.4

    # TRUE world rule: heavy AND not in {balsa, foam}
    is_sink = is_heavy and (material not in ["balsa", "foam"])

    # Noise
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


def generate_dataset(n=300):
    return [generate_object(i) for i in range(n)]


def train_test_split(data, train_ratio=0.7):
    random.shuffle(data)
    n_train = int(len(data) * train_ratio)
    return data[:n_train], data[n_train:]


def run_experiment():
    print("=== Multi-Exception Pattern Test ===\n")
    random.seed(123)

    data = generate_dataset(n=300)
    train, test = train_test_split(data, train_ratio=0.7)

    print(f"Total: {len(data)} (train={len(train)}, test={len(test)})")

    brain = Cortex()

    print("\n[TRAIN] Absorbing training data for 'sink'...")
    brain.absorb_memory(train, target_label="sink")

    print("\n[TEST] Evaluating...")

    tp = tn = fp = fn = 0
    sample_cases = []

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

        if len(sample_cases) < 12:
            sample_cases.append((obj, res))

    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total else 0.0

    print(f"\n[RESULTS]")
    print(f"Accuracy: {acc*100:.2f}%  (TP={tp}, TN={tn}, FP={fp}, FN={fn})")

    print("\n[EXPLANATIONS - SAMPLE]")
    for obj, res in sample_cases:
        print("\n----------------------------------------")
        print(f"id={obj['id']}, material={obj['material']}, is_heavy={obj['is_heavy']}")
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
