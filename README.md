[![PyPI](https://img.shields.io/pypi/v/cortex-omega?color=blue)](https://pypi.org/project/cortex-omega/)

# CORTEX-Ω
### The Epistemic Inference Engine for Python.
*No prompts. No hallucinations. Just logic.*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🛑 The Problem
Modern AI has a fatal flaw: **Catastrophic Forgetting.**
If you retrain a Neural Network to learn a new regulation, it often silently forgets the old ones. Furthermore, in high-entropy environments (noisy data), LLMs confuse correlation with causation, leading to hallucinations.

## ⚡ The Solution: Cortex
**Cortex** is not a Neural Network. It is a **Neuro-Symbolic Kernel** that decouples memory from reasoning. It utilizes **Stochastic Logic Annealing** to crystallize truth from chaos.

### Key Capabilities
* **High-Dimensional Noise Filtering:** Automatically ignores irrelevant variables (entropy regularization).
* **Non-Monotonic Plasticity (The "Kill Switch"):** Can unlearn a fundamental axiom in a single cycle without destabilizing the rest of the system.
* **Hierarchical Logic:** Prioritizes specific exceptions over general rules ("David vs. Goliath" protocol).

---

## 📦 Installation

```bash
pip install cortex-omega
```

For detailed method signatures and configuration options, see the [API Reference](API_REFERENCE.md).

---

## 🚀 Quick Start: The 5-Line Magic

Cortex ingests raw, dirty data and extracts pure logic.

```python
from cortex_omega import Cortex

# 1. Initialize the Kernel
brain = Cortex()

# 2. Absorb Dirty Data (No ETL required)
# Cortex automatically detects that 'location' is noise and 'amount' is significant.
brain.absorb("financial_chaos_logs.csv")

# 3. Query the Logic
# "Is a guest user with a heavy transaction fraudulent?"
result = brain.query(type="guest", amount="heavy")

print(f"Prediction: {result.prediction}") # Output: True
print(f"Logic Trace: {result.axiom}")      # Output: Rule_Gen_2 (Conf: 1.0)
```

---

## 📉 Evidence: Real-Time Plasticity

Unlike LLMs, Cortex guarantees **Isolation**. When the context changes (Kill Switch activated), the obsolete rule crashes to 0% confidence, while orthogonal knowledge remains at 100%.

![Real-Time Plasticity Graph](cortex_live_proof.png)

*Figure 1: Real-time execution trace showing the "Blue Rule" stability vs. the "Red Rule" collapse upon context shift.*

---

## 📚 Advanced Usage

### The "David vs. Goliath" Protocol
Cortex naturally handles logic conflicts where a specific exception contradicts a general rule.

```python
# General Rule: Heavy objects sink (Confidence 1.0)
brain.absorb_memory(heavy_objects_data, target="sink")

# Exception: Balsa wood is heavy but floats
brain.absorb_memory(balsa_wood_data, target="float")

# Query
print(brain.query(material="iron"))  # -> Sinks (General)
print(brain.query(material="balsa")) # -> Floats (Specific Exception)
```

---

## 🤝 Contributing
Cortex is an open-source epistemic engine. We welcome contributions in:
* Entropy Optimization Algorithms
* Graph Theory Visualizers
* New Data Ingestors

## 📄 License
MIT License. Copyright (c) 2025 Ivan Lozada.
