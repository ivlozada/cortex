# Release Notes

## v1.2.2: The "David & Goliath" Patch (2025-11-28)
**Robustness & Stability Update**

This release focuses on critical bug fixes and engine robustness, specifically addressing conflict resolution and noise tolerance.

### 🐛 Bug Fixes
- **Conflict Resolution**: Fixed a critical issue where boolean values (`True`/`False`) were misinterpreted as variables, preventing specific exceptions from overriding general rules (The "David vs. Goliath" bug).
- **Noise Robustness**: Improved `HypothesisGenerator` to prioritize structural properties (shape, material) over transient ones (color), preventing overfitting to noise.
- **Explanation Logic**: Fixed a bug where explanations could incorrectly cite negative rules for positive predictions.

### 🛠 Engineering
- **Test Suite**: Added comprehensive tests for robustness (`test_robustness.py`), logic (`test_logic.py`), and serialization (`test_serialization.py`).
- **Exceptions**: Introduced a unified exception hierarchy in `cortex_omega.core.errors`.

---
: Cortex-Omega v1.2.1

## The "Clean Launch" Patch

This patch reduces the default logging verbosity to ensure a professional, silent launch experience.

### 🛠 Improvements

- **Silent by Default**: The engine now defaults to `WARNING` logging level. The "Matrix code" debug output is suppressed unless explicitly enabled.
- **Logging Integration**: Replaced all `print` statements with proper `logging` calls (`info`, `debug`, `warning`).

---

# Release Notes: Cortex-Omega v1.2.0

## The "Glass Box" Update

This release transforms Cortex-Omega from a powerful but opaque inference engine into a fully verifiable "Glass Box" system.

### 🌟 New Features

#### Epistemic Traceback (The "Receipt")
- **Full Derivation Chains**: The engine now explains *why* it reached a conclusion by providing the full chain of logical steps.
- **`Proof` Object**: The `InferenceResult` now contains a structured `proof` object, which details every rule and fact used in the derivation.
- **Recursive Backtracking**: The inference engine now tracks the provenance of every derived fact, allowing for complete reconstruction of the logic path.

### 🛠 API Changes

- **`InferenceResult.proof`**: Added a new field to access the derivation chain.
- **`InferenceResult.explanation`**: Renamed from `axiom` to `explanation` for clarity (v1.1.1 fix included).
- **`Cortex.query()`**: Now populates the `proof` field when a prediction is made.

### 🐛 Bug Fixes

- Fixed `ImportError` for `MotifLibrary` (v1.1.1).
- Fixed missing explanation attribute in `InferenceResult`.

---

* "No prompts. No hallucinations. Just logic." *
