[![PyPI](https://img.shields.io/pypi/v/cortex-omega?color=blue)](https://pypi.org/project/cortex-omega/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/ivlozada/cortex-omega/actions/workflows/test.yml/badge.svg)](https://github.com/ivlozada/cortex-omega/actions)

# CORTEX-Ω
### The Epistemic Inference Engine for Python.
*No prompts. No hallucinations. Just logic.*

**Cortex-Omega is a neuro-symbolic pattern learner that induces human-readable rules (with exceptions) from tabular data, with Bayesian robustness to noise.**

---

## ⚡ Quick Start

```bash
pip install cortex-omega
```

```python
from cortex_omega import Cortex

# 1. Initialize (Robust Mode by default)
brain = Cortex()

# 2. Teach (Pattern: Heavy things sink, but Balsa is an exception)
brain.absorb_memory([
    {"id": "iron", "heavy": "true", "material": "iron", "sink": "true"},
    {"id": "stone", "heavy": "true", "material": "stone", "sink": "true"},
    {"id": "balsa", "heavy": "true", "material": "balsa", "sink": "false"}, # Exception!
], target_label="sink")

# 3. Query
result = brain.query(heavy="true", material="balsa", target="sink")

print(f"Prediction: {result.prediction}")   # -> False
print(f"Confidence: {result.confidence:.2f}") # -> 1.00
print(f"Explanation: {result.explanation}") # -> R_exception: ¬sink(X) :- material(X, balsa)
```

### 🛡️ Epistemic Safety
Cortex refuses to guess. If it has no knowledge to answer a query, it raises an `EpistemicVoidError` instead of hallucinating.

```python
try:
    brain.query(id="unknown_obj", target="sink")
except EpistemicVoidError:
    print("I don't know enough to answer that.")
```

### 🔍 Theory Inspector
Want to see the actual logic the brain learned?
```python
for rule in brain.inspect_rules(target="sink"):
    print(rule)
# Output:
# sink(X) :- heavy(X), ¬material(X, balsa) [0.96]
```

### ⚙️ Profiles
Cortex comes with pre-tuned profiles for different use cases:

```python
# Default: Robust Pattern Learner
# Balances generalization with exception handling. Best for noisy real-world data.
brain = Cortex()

# Strict: Legal / Logic Profile
# Prioritizes specific exceptions over general rules. A single counter-example can override a law.
# Best for compliance, policy, and safety-critical systems.
legal_brain = Cortex(mode="strict")
```

#### Mode Comparison

| Mode | Behavior | Use Case |
| :--- | :--- | :--- |
| **Robust** | Bayesian, noise-tolerant. Requires multiple counter-examples to override a rule. | Noisy tabular data, fraud detection, ops. |
| **Strict** | Zero Tolerance. A single counter-example overrides a general rule (Kill Switch). | Policy enforcement, legal compliance, safety. |

### 🔧 Advanced Config
You can fine-tune the engine's cognitive biases:

```python
brain = Cortex(
    mode="strict",
    feature_priors={"material": 5.0},   # Prefer 'material' explanations over others
    noise_model={"false_positive": 0.1}, # Expect 10% sensor noise
    debug=True # Enable verbose logging and tracing
)
```

*   **`feature_priors`**: Guide the engine towards specific causal factors.
*   **`noise_model`**: Adjust tolerance for false positives/negatives.
*   **`sensitivity`**: Control the "imagination" (willingness to accept weak correlations).

---

### 1. The "Killer Demo" (Project Prometheus)
Simulates an insider threat scenario where an admin account is compromised. Cortex-Omega learns:
*   **Baseline**: Users accessing secrets = Threat. Admins accessing secrets = Safe.
*   **Attack Vector**: Anyone touching the 'honeypot_db' is a threat (overriding Admin status).
*   **Result**: It catches the compromised admin in real-time.
Run it: `python examples/10_killer_demo_prometheus.py`

### 2. Project Cassandra (Social Virality Simulator)
**New in v2.0.0**: Demonstrates the "Neuro-Symbolic Gap".
*   **Phase 1 (Robust Learning)**: Learns noisy viral patterns (e.g., "Mega Influencers usually go viral") despite exceptions.
*   **Phase 2 (Strict Enforcement)**: Injects a "Truth Axiom" (`NOT_is_viral :- fact_check(false)`) to suppress misinformation.
*   **Result**: It correctly predicts viral hits while strictly banning false content, even from influencers.
Run it: `python examples/11_project_cassandra.py`

### 3. Project Gödel (Recursive Arithmetic)
**New in v2.0.0**: Demonstrates the ability to learn recursive rules and structured terms (Peano Arithmetic).
*   **Goal**: Learn addition `add(X, Y, Z)` from examples like `0+0=0`, `0+1=1`, `1+1=2`.
*   **Representation**: Uses nested terms `s(s(zero))` instead of flat strings.
*   **Result**: Learns the recursive rule `add(X, s(Y), s(Z)) :- add(X, Y, Z)` and generalizes to unseen numbers (e.g., `4+1=5`).
Run it: `python examples/12_recursive_arithmetic.py`

## 📦 Installations).

---

## 🌟 Why Cortex?

Modern AI has a fatal flaw: **Catastrophic Forgetting.** Neural Networks struggle to learn specific exceptions without unlearning general rules.

Cortex is different. It uses **Stochastic Logic Annealing** to crystallize truth from chaos.

### Key Capabilities

*   **Rule Induction with Exceptions**: Learns "David vs. Goliath" logic (specific rules override general ones).
*   **Recursive Term Support (v2.0)**: Handles nested structures (e.g., `s(s(zero))`) and learns recursive rules (Project Gödel).
*   **Glass-Box Explanations**: Every prediction comes with a logical proof trace.
*   **Robustness to Noise**: Bayesian scoring filters out spurious correlations and confounders.
*   **Clean Python API**: Designed for engineers, not just researchers.
*   **Clean Python API**: Designed for engineers, not just researchers.
*   **Scikit-Learn Integration**: Drop-in `CortexClassifier` for ML pipelines.
*   **Pluggable Strategies**: Extensible architecture for adding new reasoning capabilities (see [`docs/strategies.md`](docs/strategies.md)).

See [`docs/SPEC_kernel.md`](docs/SPEC_kernel.md) for the kernel specification.

---

## 📚 Official Demos

*   **[`examples/01_binary_classification.py`](examples/01_binary_classification.py)**: "Hello World" - Simple rule learning.
*   **[`examples/02_relational_grandparent.py`](examples/02_relational_grandparent.py)**: Relational learning (A is parent of B, B is parent of C -> A is grandparent of C).
*   **[`examples/03_robust_noisy_labels.py`](examples/03_robust_noisy_labels.py)**: Robustness demo - Learning signal amidst noise.
*   **[`examples/04_strobe_light_protocol.py`](examples/04_strobe_light_protocol.py)**: High-frequency adaptability test.

## 💡 Examples

### 1. Socrates (Logical Deduction)
```python
brain.add_rule("human(X) :- greek(X)")
brain.add_rule("mortal(X) :- human(X)")
brain.absorb_memory([{"id": "socrates", "greek": "true"}], target_label="greek")
print(brain.query(id="socrates", target="mortal").prediction) # -> True
```

### 2. David vs. Goliath (Conflict Resolution)
Cortex learns that "Birds fly", but "Penguins are birds that do not fly".
```python
brain.absorb_memory([
    {"type": "sparrow", "bird": "true", "fly": "true"},
    {"type": "eagle", "bird": "true", "fly": "true"},
    {"type": "penguin", "bird": "true", "fly": "false"},
], target_label="fly")
```

### 3. Confounder Trap (Causal Discovery)
In a noisy dataset where "Red" correlates with "Sink" 90% of the time, but "Heavy" is the true cause, Cortex correctly identifies "Heavy" as the causal feature using Information Gain and Stability Analysis.

### 4. Pattern Learning Demo
We include a golden test script that demonstrates end-to-end learning of a "Heavy things sink, except balsa" rule from synthetic data.

```bash
python3 test_pattern_learning.py
# Validates world-rule + exception learning with 100% accuracy.
```

The engine discovers:
```prolog
sink(X) :- heavy(X), ¬material(X, balsa).
```

---

## 📊 Performance & Benchmarks

| Task | Accuracy | Noise Rate | Notes |
| :--- | :--- | :--- | :--- |
| **Pattern Learning** | 100% | 0% | Perfect recovery of synthetic rules. |
| **Noisy World** | ~90% | 20% | Resilient to label flips and feature noise. |
| **Confounder Trap** | ~89% | N/A | Successfully ignores spurious "Color" correlation. |
| **Multi-Exception** | ~89% | N/A | Handles multiple overlapping exception layers. |

*Benchmarks run on synthetic datasets. See `tests/benchmarks/` for details.*

---

## ⚠️ Limitations

*   **Not for Perception**: Cortex is a reasoning engine, not a perception engine. Use a Neural Network to convert images/audio into symbolic features first.
*   **Tabular/Symbolic Data**: Best suited for structured data (JSON, CSV, SQL).
*   **Complexity**: Current implementation scales to ~10k rules.

---

## 🤝 Contributing

We welcome contributions! Please see `ROADMAP.md` for future plans.

## 📄 License

Cortex-Omega is released under the MIT License. See the [LICENSE](LICENSE) file for details.
