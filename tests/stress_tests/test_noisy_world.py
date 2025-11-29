"""
Test: Noisy / Contradictory Labels

True world rule:
    is_sink_clean = is_heavy and (material != "balsa")

But we inject label noise:
    With small probability p_noise, we FLIP the is_sink label.

We want to see:
  - Does Cortex still learn something close to the true rule?
  - Is test accuracy high (but not perfect)?
  - Do confidences drop (no more 0.99 everywhere)?
"""

import random
from cortex_omega import Cortex

MATERIALS = ["iron", "lead", "stone", "balsa", "plastic", "wood", "foam"]
COLORS = ["red", "blue", "green", "yellow"]
SHAPES = ["sphere", "cube", "block", "sheet"]


def generate_noisy_object(idx: int, p_noise: float):
    material = random.choice(MATERIALS)

    # base heaviness rule
    if material in ["iron", "lead", "stone"]:
        is_heavy = True
    elif material == "balsa":
        is_heavy = True
    else:
        is_heavy = random.random() < 0.4

    is_sink_clean = is_heavy and (material != "balsa")

    # flip label with probability p_noise
    if random.random() < p_noise:
        is_sink = not is_sink_clean
    else:
        is_sink = is_sink_clean

    return {
        "id": f"obj_{idx}",
        "material": material,
        "is_heavy": is_heavy,
        "is_sink": is_sink,
        "is_sink_clean": is_sink_clean,
        "color": random.choice(COLORS),
        "shape": random.choice(SHAPES),
        "random_id": random.randint(1, 10_000),
    }


def generate_dataset(n=300, p_noise=0.1):
    return [generate_noisy_object(i, p_noise=p_noise) for i in range(n)]


def train_test_split(data, train_ratio=0.7):
    random.shuffle(data)
    n_train = int(len(data) * train_ratio)
    return data[:n_train], data[n_train:]


def run_experiment(p_noise=0.1):
    print(f"=== Noisy World Test (noise={p_noise*100:.1f}%) ===\n")
    random.seed(777)

    data = generate_dataset(n=300, p_noise=p_noise)
    train, test = train_test_split(data, train_ratio=0.7)

    print(f"Total={len(data)}, Train={len(train)}, Test={len(test)}")

    brain = Cortex()

    print("\n[TRAIN] Absorbing noisy labels for 'sink'...")
    brain.absorb_memory(train, target_label="sink")

    print("\n[TEST] Evaluating on test with noisy labels...")

    tp = tn = fp = fn = 0
    sample_cases = []

    for obj in test:
        true_label = obj["is_sink"]  # noisy label

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
    acc_noisy = (tp + tn) / total if total else 0.0

    # Now also measure against the *clean* underlying label:
    tp_c = tn_c = fp_c = fn_c = 0
    for obj, res in sample_cases:
        true_clean = obj["is_sink_clean"]
        pred_label = bool(res.prediction)
        if pred_label and true_clean:
            tp_c += 1
        elif not pred_label and not true_clean:
            tn_c += 1
        elif pred_label and not true_clean:
            fp_c += 1
        elif not pred_label and true_clean:
            fn_c += 1
    total_c = tp_c + tn_c + fp_c + fn_c
    acc_clean_est = (tp_c + tn_c) / total_c if total_c else 0.0

    print(f"\n[RESULTS]")
    print(f"Accuracy vs NOISY labels (test): {acc_noisy*100:.2f}%")
    print(f"(On sample only) Accuracy vs CLEAN underlying rule: {acc_clean_est*100:.2f}%")

    print("\n[EXPLANATIONS - SAMPLE TEST OBJECTS]")
    for obj, res in sample_cases:
        print("\n----------------------------------------")
        print(f"id={obj['id']}, material={obj['material']}, is_heavy={obj['is_heavy']}")
        print(f"TRUE (noisy) is_sink      = {obj['is_sink']}")
        print(f"TRUE (clean) is_sink_clean = {obj['is_sink_clean']}")
        print(f"Prediction   = {res.prediction} (conf={res.confidence:.2f})")
        print(f"Explanation  = {res.explanation}")
        if res.proof is not None:
            print(f"Proof        = {res.proof}")
        else:
            print("Proof        = (no supporting rule; defaulted to negative)")

    print("\n=== Done. ===")


if __name__ == "__main__":
    run_experiment(p_noise=0.15)  # 15% flipped labels
