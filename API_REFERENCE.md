# CORTEX-Ω API Reference

This document details the public API surface for the `cortex-omega` package.

---

## 1. Core Engine

### `class cortex_omega.Cortex`
The main entry point for the Epistemic Inference Engine.

#### `__init__(self, sensitivity: float = 0.1, mode: str = "robust", priors: dict = None, noise_model: dict = None, plasticity: dict = None, feature_priors: dict = None)`
Initializes a new Cortex kernel instance.

* **Parameters:**
    * `sensitivity` (*float*): Controls the entropy penalty ($\lambda$).
        * Values closer to `0.0` make the engine "imaginative" (accepts weak correlations).
        * Values closer to `1.0` make the engine "skeptical" (requires strong causality).
        * *Default:* `0.1` (Recommended for dirty CSVs).
    * `mode` (*str*): Operating mode.
        * `"robust"` (Default): Bayesian, noise-tolerant. Requires multiple counter-examples to override a rule.
        * `"strict"`: Zero Tolerance. A single counter-example overrides a general rule.
    * `priors` (*dict, optional*): Bayesian priors for rule generation (e.g., `{"rule_base": 0.5}`).
    * `noise_model` (*dict, optional*): Expected noise rates (e.g., `{"false_positive": 0.05}`).
    * `plasticity` (*dict, optional*): Parameters for rule retention and memory limits.
    * `feature_priors` (*dict, optional*): Causal hints for feature selection (e.g., `{"color": 0.1, "material": 0.9}`).
    * `random_seed` (*int, optional*): Seed for deterministic behavior (CI/Testing).

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

#### `add_rule(self, rule_str: str)`
Injects a human-written rule into the theory.

* **Parameters:**
    * `rule_str` (*str*): Prolog-style string.
        * Example: `"mortal(X) :- man(X)"`
        * Example: `"fraud(X) :- transaction(X, amount, V), V > 10000"`
* **Raises:** `RuleParseError` if the string is invalid.

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

#### `save_brain(self, path: str)`
Serializes the full brain (theory + memory) to disk via pickle.

* **Parameters:**
    * `path` (*str*): Path to the output file.

#### `load_brain(path: str) -> Cortex`
Static method. Loads a previously saved brain.

* **Parameters:**
    * `path` (*str*): Path to the input file.
* **Returns:** Loaded `Cortex` instance.

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
    * `prediction` (*bool | None*): The logical conclusion (`True` / `False`). Returns `None` if the system has insufficient knowledge (epistemic void).
    * `confidence` (*float*): A score between `0.0` and `1.0` indicating the strength of the axiom used.
    * `explanation` (*str | None*): The string representation of the crystallized rule used for deduction.
        * Example: `"fraud(X) :- mass(X, heavy), type(X, guest)"`
    * `proof` (*Any | None*): A structured proof object containing the derivation trace.

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
