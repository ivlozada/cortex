# Cortex-Omega Examples

This document provides a detailed walkthrough of the example scripts in the `examples/` directory. Each example demonstrates a core capability of the Cortex-Omega engine.

## 01. Binary Classification (Hello World)
**Script:** `examples/01_binary_classification.py`

**Concept:** Teaches the system to classify objects based on a single feature, ignoring irrelevant ones.

**Scenario:**
- **Rule:** Red objects are "hazardous".
- **Data:** A mix of red and blue objects.
- **Goal:** Learn `hazardous(X) :- color(X, red)`.

**Pseudo-Code:**
```python
cortex = Cortex()
cortex.absorb([
    {"color": "red", "hazardous": True},
    {"color": "blue", "hazardous": False}
], target="hazardous")

result = cortex.query(color="red", target="hazardous")
# Expect: True
```

**Expected Output:**
```
🧠 Cortex Initialized.
📚 Absorbing 4 training examples...
🔮 Predictions:
  Object test1 (red): Hazardous? True
    Reason: hazardous(X) :- color(X, red)
  Object test2 (blue): Hazardous? False
```

---

## 02. Relational Learning (Grandparent)
**Script:** `examples/02_relational_grandparent.py`

**Concept:** Demonstrates learning multi-hop relational rules.

**Scenario:**
- **Facts:** Abe -> Homer -> Bart.
- **Goal:** Learn that Abe is Bart's grandparent because of the chain of parent relationships.

**Pseudo-Code:**
```python
cortex.facts.add("parent", ("abe", "homer"))
cortex.facts.add("parent", ("homer", "bart"))

cortex.absorb([
    {"id": "abe", "is_grandparent": True},
    {"id": "bart", "is_grandparent": False}
], target="is_grandparent")

# Query
result = cortex.query(id="abe", target="is_grandparent")
# Expect: True
```

**Expected Output:**
```
📚 Absorbing family tree and examples...
🔮 Predictions:
  Grampa Smurf: True
    Reason: is_grandparent(X) :- parent(X, Z), parent(Z, Y)
```

---

## 03. Robustness to Noisy Labels
**Script:** `examples/03_robust_noisy_labels.py`

**Concept:** Demonstrates the engine's ability to ignore spurious correlations (confounders) and focus on robust, causal features.

**Scenario:**
- **Signal:** "Square" objects are targets.
- **Noise:** "Red" objects are *mostly* targets (80% correlation), but not always.
- **Goal:** Learn `target(X) :- shape(X, square)` and ignore `color(X, red)`.

**Pseudo-Code:**
```python
cortex = Cortex(mode="robust")
cortex.absorb([
    {"shape": "square", "color": "red", "target": True}, # Confounder
    {"shape": "circle", "color": "red", "target": False}, # Counter-example for color
    {"shape": "square", "color": "blue", "target": True}  # Proof for shape
], target="target")

# Query Red Circle
result = cortex.query(shape="circle", color="red", target="target")
# Expect: False (Correctly ignores Red)
```

**Expected Output:**
```
🔮 Predictions:
  Object test_blue_square (square, blue): Target? True
    Reason: target(X) :- shape(X, square)
  Object test_red_circle (circle, red): Target? False
```

---

## 04. Strobe Light Protocol (Adaptability)
**Script:** `examples/04_strobe_light_protocol.py`

**Concept:** Tests high-frequency adaptability (plasticity). The "ground truth" changes rapidly (Red is True -> Red is False -> Red is True).

**Scenario:**
- **Cycle 0-10:** Red -> Glow
- **Cycle 10-20:** Red -> No Glow
- **Goal:** Cortex should adapt its confidence in real-time.

**Expected Output:**
```
[TEST] Starting 100 cycles of rapid concept drift...
   Cycle 000: Reality is ON  | Cortex Confidence: 0.90
   Cycle 010: Reality is OFF | Cortex Confidence: 0.00
   Cycle 020: Reality is ON  | Cortex Confidence: 0.85
[SUCCESS] Strobe graph generated: cortex_strobe_proof.png
```

---

## 12. Recursive Arithmetic (Project Gödel)
**Script:** `examples/12_recursive_arithmetic.py`

**Concept:** Demonstrates learning recursive rules and structured terms (Peano Arithmetic).

**Scenario:**
- **Representation:** Numbers are nested terms: `0` -> `zero`, `1` -> `s(zero)`, `2` -> `s(s(zero))`.
- **Goal:** Learn addition `add(X, Y, Z)` from examples.
- **Learned Logic:**
    1. Base Case: `add(X, zero, X)`
    2. Recursive Step: `add(X, s(Y), s(Z)) :- add(X, Y, Z)`

**Pseudo-Code:**
```python
# Teach 0+0=0, 0+1=1, 1+0=1, 1+1=2...
cortex.absorb(examples)

# Query 4+1=5 (Unseen)
# Cortex uses the recursive rule to decompose 4+1 -> 4+0 -> 4, then rebuilds to 5.
result = cortex.query(add(s(s(s(s(zero)))), s(zero), Z))
# Expect: Z = s(s(s(s(s(zero)))))
```

**Expected Output:**
```
📜 Learned Rules:
  - add(X, zero, X)
  - add(X, s(Y), s(Z)) :- add(X, Y, Z)

🧮 Testing Generalization (4+1=5)...
  4 + 1 = 5? -> True
```
