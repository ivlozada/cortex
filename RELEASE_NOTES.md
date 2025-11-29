# Release Notes: Cortex-Omega v1.2.1

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
