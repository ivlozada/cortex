from cortex_omega import Cortex
import random

def generate_biased_training(n=200):
    """
    Early phase: Red & Square -> True (both correlated)
    Later phase: Red stops working, Square remains causal.
    """
    data = []
    for i in range(n // 2):
        data.append({"id": f"early_{i}",
                     "color": "red",
                     "shape": "square",
                     "glows": True})

    for i in range(n // 4):
        data.append({"id": f"late_bad_{i}",
                     "color": "red",
                     "shape": "circle",
                     "glows": False})
    for i in range(n // 4):
        data.append({"id": f"late_good_{i}",
                     "color": "blue",
                     "shape": "square",
                     "glows": True})
    random.shuffle(data)
    return data

def test_confounder_invariance():
    random.seed(123)
    brain = Cortex()
    data = generate_biased_training()
    brain.absorb_memory(data, target_label="glows")

    # Test 1: causal feature (shape='square') generalizes across colors
    ok_square = 0
    total_square = 0
    for color in ["red", "blue", "green"]:
        res = brain.query(color=color, shape="square", target="glows")
        if res.prediction is True:
            ok_square += 1
        total_square += 1

    # Test 2: non-causal confounder (color='red') should *not* force glows=True
    res_red_circle = brain.query(color="red", shape="circle", target="glows")

    assert ok_square == total_square, "Square rule not stable across colors"
    assert res_red_circle.prediction is False, "Engine still believes 'red -> glow'"
