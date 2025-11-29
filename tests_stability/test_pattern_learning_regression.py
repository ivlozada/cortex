import random
from cortex_omega import Cortex

MATERIALS = ["iron", "lead", "stone", "balsa", "plastic", "wood", "foam"]

def generate_object(idx: int):
    import random
    material = random.choice(MATERIALS)

    if material in ["iron", "lead", "stone"]:
        is_heavy = True
    elif material == "balsa":
        is_heavy = True     # forced heavy exception
    else:
        is_heavy = random.random() < 0.3

    is_sink = is_heavy and (material != "balsa")

    return {
        "id": f"obj_{idx}",
        "material": material,
        "is_heavy": is_heavy,
        "is_sink": is_sink,
    }

def generate_dataset(n=400):
    return [generate_object(i) for i in range(n)]

def train_test_split(data, train_ratio=0.7):
    random.shuffle(data)
    n_train = int(len(data) * train_ratio)
    return data[:n_train], data[n_train:]

def test_pattern_learning_clean_world():
    random.seed(42)
    data = generate_dataset(400)
    train, test = train_test_split(data, 0.7)

    brain = Cortex()
    brain.absorb_memory(train, target_label="sink")

    correct = 0
    for obj in test:
        res = brain.query(
            material=obj["material"],
            is_heavy=obj["is_heavy"],
            target="sink",
        )
        pred = bool(res.prediction)
        if pred == obj["is_sink"]:
            correct += 1

    acc = correct / len(test)
    # In clean deterministic world we expect ~0.95–1.00
    assert acc >= 0.95, f"Accuracy too low: {acc:.3f}"
