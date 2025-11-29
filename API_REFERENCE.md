# CORTEX-Ω API Reference

This document details the public API surface for the `cortex-omega` package.

---

## 1. Core Engine

### `class cortex_omega.Cortex`
The main entry point for the Epistemic Inference Engine.

#### `__init__(self, sensitivity: float = 0.1)`
Initializes a new Cortex kernel instance.

* **Parameters:**
    * `sensitivity` (*float*): Controls the entropy penalty ($\lambda$).
        * Values closer to `0.0` make the engine "imaginative" (accepts weak correlations).
        * Values closer to `1.0` make the engine "skeptical" (requires strong causality).
        * *Default:* `0.1` (Recommended for dirty CSVs).

#### `absorb(self, file_path: str, target_col: str = None)`
Ingests a raw data file (CSV/JSON), performs entropy analysis to identify noise, and updates the internal logic theory.

* **Parameters:**
    * `file_path` (*str*): Relative or absolute path to the data file.
    * `target_col` (*str, optional*): Explicitly defines the ground truth column. If `None`, Cortex attempts to auto-detect boolean or categorical targets.
* **Returns:** `None`
* **Raises:** `FileNotFoundError`, `ValueError` (if file format is unsupported).

#### `absorb_memory(self, data: list[dict], target_label: str, result: bool = True)`
Injects specific memories or rules directly into the engine without reading a file. Useful for "teaching" specific exceptions or enforcing protocols.

* **Parameters:**
    * `data` (*list[dict]*): A list of dictionaries representing the context.
        * Example: `[{'color': 'red', 'mass': 'heavy'}]`
    * `target_label` (*str*): The predicate being learned (e.g., `'fraud'`, `'glow'`).
    * `result` (*bool*): The ground truth for this observation.
* **Note:** Use this method to trigger the **Kill Switch** by providing new data that contradicts established axioms.

* **Note:** Use this method to trigger the **Kill Switch** by providing new data that contradicts established axioms.

#### `set_mode(self, mode: str)`
Dynamically switches the operating mode.

* **Parameters:**
    * `mode` (*str*): `"strict"` (Zero Tolerance) or `"robust"` (Bayesian).
* **Returns:** `None`

#### `export_rules(self, format: str = "json") -> str`
Exports learned rules to a standard format.

* **Parameters:**
    * `format` (*str*): `"json"` or `"prolog"`.
* **Returns:** String containing the exported rules.

#### `inspect_rules(self, target: str = None) -> List[Rule]`
Returns a list of crystallized rules, optionally filtered by a target predicate.

* **Parameters:**
    * `target` (*str, optional*): The predicate to filter by (e.g., `'fraud'`).
* **Returns:** List of `Rule` objects with full statistics.

#### `query(self, **kwargs) -> PredictionResult`
Queries the crystallized logic engine for a prediction based on partial evidence.

* **Parameters:**
    * `**kwargs`: Arbitrary keyword arguments representing the known facts.
        * Example: `brain.query(color='red', mass='heavy')`
* **Returns:** `PredictionResult` object.

---

## 2. Data Structures

### `class cortex_omega.api.client.PredictionResult`
The object returned by a query, containing both the inference and the epistemic metadata.

* **Attributes:**
    * `prediction` (*bool*): The logical conclusion (`True` / `False`). Returns `None` if the system has insufficient knowledge (epistemic void).
    * `confidence` (*float*): A score between `0.0` and `1.0` indicating the strength of the axiom used.
    * `axiom` (*str*): The string representation of the crystallized rule used for deduction.
        * Example: `"fraud(X) :- mass(X, heavy), type(X, guest)"`

### `class cortex_omega.core.rules.Rule`
Represents a crystallized logical pattern.

* **Attributes:**
    * `head` (*Literal*): The conclusion of the rule (e.g., `fraud(X)`).
    * `body` (*List[Literal]*): The conditions (e.g., `[amount(X, heavy), type(X, guest)]`).
    * `confidence` (*float*): Bayesian belief score (0.0 - 1.0).
    * `fires_pos` (*int*): Number of times the rule fired correctly (True Positive).
    * `fires_neg` (*int*): Number of times the rule fired incorrectly (False Positive).
    * `reliability` (*float*): `fires_pos / (fires_pos + fires_neg)`.
    * `complexity` (*int*): MDL complexity score (length of body + 1).

---

## 3. Configuration

### `class cortex_omega.core.engine.KernelConfig`
Advanced configuration for the inner Structural Gradient Descent loop.

* **Attributes:**
    * `lambda_complexity` (*float*): Cost of adding a new variable to a rule.
    * `min_confidence` (*float*): Threshold for promoting a hypothesis to an Axiom. Default `0.85`.
    * `max_rules` (*int*): Hard limit on memory slots to prevent combinatorial explosion. Default `1000`.

---

## 4. Error Handling

* **`EpistemicVoidError`**: Raised when querying a concept the brain has never encountered.
* **`ConflictWarning`**: Logged when "David vs. Goliath" logic overrides a general rule (standard behavior, not a crash).

---

## 5. Integrations

### `class cortex_omega.api.sklearn.CortexClassifier`
A Scikit-Learn compatible wrapper for Cortex Omega.

#### `__init__(self, target_label: str, sensitivity: float = 0.1, mode: str = "robust")`
#### `fit(self, X, y=None)`
#### `predict(self, X) -> np.ndarray`
#### `predict_proba(self, X) -> np.ndarray`
