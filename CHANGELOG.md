# Changelog

All notable changes to this project will be documented in this file.

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
