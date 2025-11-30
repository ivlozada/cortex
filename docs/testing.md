# Cortex-Omega Testing Guide

This document outlines the testing strategy for Cortex-Omega and how to run the test suite.

## Test Structure

The tests are organized into two main directories:

1.  **`tests/`**: Core unit tests and functional tests.
    *   `test_robustness.py`: Verifies resilience to noise and correct feature selection.
    *   `test_mdl.py`: Tests the Minimum Description Length (MDL) scoring logic.
    *   `test_logic.py`: Tests the core inference engine and rule parsing.
    *   `test_pattern_learning_golden.py`: End-to-end "Golden Set" tests.

2.  **`tests_stability/`**: Integration and stability tests.
    *   `test_david_vs_goliath.py`: Verifies that specific exceptions override general rules.
    *   `test_stability.py`: Long-running stability checks.

## Running Tests

We use `pytest` as the test runner.

### Run All Tests
```bash
pytest
```

### Run Specific Test File
```bash
pytest tests/test_robustness.py
```

### Run Tests Matching a Keyword
```bash
pytest -k "david"
```

### Run with Verbose Output
```bash
pytest -v
```

## Key Test Scenarios

### Robustness Test (`tests/test_robustness.py`)
Ensures the engine doesn't overfit to noisy features.
- **Input:** Data where Color is 80% correlated with Target, but Shape is 100% correlated.
- **Pass Condition:** Engine learns Rule(Shape) and ignores Rule(Color).

### David vs. Goliath (`tests_stability/test_david_vs_goliath.py`)
Ensures specific rules override general ones.
- **Input:** "Heavy things sink" (General) + "Balsa wood floats" (Exception).
- **Pass Condition:**
    - `query(iron)` -> Sinks (General Rule)
    - `query(balsa)` -> Floats (Exception Rule)

### Strobe Light Protocol (`examples/04_strobe_light_protocol.py`)
Tests real-time adaptability.
- **Input:** Ground truth flips every N cycles.
- **Pass Condition:** Confidence score tracks the ground truth signal with minimal lag.
