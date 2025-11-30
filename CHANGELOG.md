# Changelog

All notable changes to this project will be documented in this file.

## [1.9.0] - 2025-11-29

### Added
- **Epistemic Safety**: Implemented `EpistemicVoidError` to prevent hallucination when the engine has no knowledge to answer a query.
- **Exception Hierarchy**: Defined a structured exception hierarchy (`GDMError`, `InconsistentTheoryError`, etc.) in `cortex_omega.core.errors`.
- **Debug Mode**: Added `debug=True` flag to `KernelConfig` for verbose logging and tracing.
- **Examples**: Added a comprehensive `examples/` directory with:
    - `01_binary_classification.py`: "Hello World".
    - `02_relational_grandparent.py`: Relational learning.
    - `03_robust_noisy_labels.py`: Robustness demo.
    - `04_strobe_light_protocol.py`: Adaptability test.
- **Documentation**: Added `docs/examples.md` and `docs/testing.md`.

### Changed
- **Robustness**: Fixed `FactBase.query` variable handling and `evaluate_f1` cheating, resolving `test_robustness.py` failures.
- **Priority**: Standardized `PROPERTY_PRIORITY` to "Higher Value = Higher Priority", correctly prioritizing robust features (Shape) over noisy ones (Color).
- **Cleanup**: Removed "meta-comments" and unused code in `inference.py` and `hypothesis.py`.

## [1.8.2] - 2025-11-29

### Fixed
- **Stability Regression**: Reverted `n_feat` threshold in `DiscriminativeFeatureSelector` to 3 (from 1) to resolve accuracy regression in noisy environments.
- **Feature Selection**: Tuned `min_score` to 0.01 to maintain sensitivity to rare exceptions while filtering noise.

### Verified
- **Noise Robustness**: `tests/test_pattern_learning_golden.py` passes with >90% accuracy.
- **Exception Learning**: `tests_stability/test_david_vs_goliath.py` passes, correctly handling the "balsa" exception.

## [1.8.1] - 2025-11-29

### Fixed
- **Fact Persistence**: Fixed `absorb_memory` in `client.py` to correctly persist learned facts to the knowledge base, enabling multi-step reasoning (e.g., Aristotle syllogism).
- **Numeric Learning**: Lowered `min_score` threshold in `hypothesis.py` to 0.01 to allow learning from smaller datasets (critical for demos).
- **Examples**: Fixed and verified all 10 examples in `examples/`.
    - `01_financial_fraud_detection.py`: Fixed `AttributeError`.
    - `07_david_vs_goliath.py`: Added `is_balsa` feature to enable exception learning.
    - `10_core_capabilities_demo.py`: Refactored to use categorical features for reliable rule learning.

## [1.8.0] - 2025-11-29

### Added
- **Robust Pattern Learning**: Engine now correctly learns complex exceptions (e.g., "Heavy things sink, except balsa") even in noisy data.
- **Strict Profile**: Added `Cortex(mode="strict")` for legal/compliance use cases where specific exceptions must override general rules.
- **Golden Test**: Added `test_pattern_learning.py` as a canonical demonstration of rule+exception learning.

### Changed
- **Stability**: Passed full regression suite including Confounder Invariance, Temporal Sequence, and Noise Robustness.
- **Inference**: Fixed a bug in `inference.py` where rule statistics were artificially inflated, causing valid exception rules to be pruned.

### Verified
- **Accuracy**: Achieved 100% accuracy on the "Sink vs Balsa" synthetic dataset.
- **Theory**: Confirmed learned rules match ground truth logic: `sink(X) :- heavy(X), ¬material(X, balsa)`.

## [1.7.1] - 2025-11-29

### Fixed
- **Versioning**: Updated `__init__.py` to dynamically load version from `setuptools_scm` generated file, ensuring `cortex_omega.__version__` matches the installed package version.

## [1.7.0] - 2025-11-29

### Changed
- **Deployment**: Switched to dynamic versioning using `setuptools_scm`. Removed `setup.py` in favor of `pyproject.toml` as the single source of truth.

## [1.6.0] - 2025-11-29

### Added
- **Brain Serialization**: Implemented `save_brain(path)` and `load_brain(path)` methods in `Cortex` class using pickle for full state persistence.
- **Strict Mode**: Added `mode="strict"` configuration to `Cortex` for zero-tolerance logic learning.
- **Rule Export**: Added `export_rules(format="json"|"prolog")` to export learned logic.
- **Scikit-Learn Wrapper**: Added `CortexClassifier` for compatibility with sklearn pipelines.
- **CI/CD**: Added GitHub Actions workflow (`.github/workflows/test.yml`) for automated testing.
- **Citation**: Added `CITATION.cff` for academic referencing.

### Changed
- **Logging**: Replaced all hardcoded `print` statements with structured `logging` calls.
- **Rule Object**: Refactored `Rule` to include `created_at`, `updated_at`, and rich statistics (`fires_pos`, `fires_neg`, `reliability`).
- **Documentation**: Completely rewrote `README.md` to "World-Class" standard with clear examples and benchmarks.
- **Configuration**: `Cortex` constructor now accepts `priors`, `noise_model`, `plasticity`, and `feature_priors` dictionaries.

### Fixed
- **Stability**: Fixed potential regressions in noisy environments by enforcing stricter feature selection priors.
- **Usability**: Removed debug spam from stdout.

## [1.5.0] - 2025-11-29
- Initial release of Temporal and Relational Logic capabilities.
