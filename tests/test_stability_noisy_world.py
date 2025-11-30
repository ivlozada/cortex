from cortex_omega import Cortex
import random

def generate_noisy_data(n=400, noise=0.15):
    data = []
    for i in range(n):
        is_heavy = random.random() < 0.5
        # true rule: heavy -> sinks
        is_sink = is_heavy
        # flip label with prob = noise
        if random.random() < noise:
            is_sink = not is_sink
        data.append({"id": f"obj_{i}",
                     "is_heavy": is_heavy,
                     "is_sink": is_sink})
    return data

def test_noisy_world_robustness():
    random.seed(2024)
    data = generate_noisy_data()
    train = data[:300]
    test  = data[300:]

    brain = Cortex()
    brain.absorb_memory(train, target_label="sink")

    correct = 0
    for obj in test:
        res = brain.query(is_heavy=obj["is_heavy"], target="sink")
        pred = bool(res.prediction)
        if pred == obj["is_sink"]:
            correct += 1

    acc = correct / len(test)
    # With 15% label noise, anything >> 0.5 shows robustness.
    assert acc >= 0.70, f"Accuracy too low under noise: {acc:.3f}"
