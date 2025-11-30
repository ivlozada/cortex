# Cortex-Omega Kernel Specification (v1.0)

## 1. The Contract

The **GDM Kernel** is the neuro-symbolic core of Cortex-Omega. Its sole purpose is to induce logical theories from examples.

**Mission:**
> "Given a set of scenes (examples) and a target predicate, produce a logical theory (set of rules) that explains the target with maximum accuracy and minimum complexity."

### 1.1 Inputs
The kernel accepts the following inputs via `fit()`:

1.  **`scenes`** (`List[Scene]`): A list of training examples.
    *   Each `Scene` contains a `FactBase` (ground truth facts) and a `target_entity`.
    *   The `FactBase` must be fully grounded (no variables).
2.  **`target_predicate`** (`str`): The name of the predicate to learn (e.g., "grandfather", "glows").
3.  **`config`** (`KernelConfig`): A configuration object containing all hyperparameters.
    *   **Invariant:** No magic numbers allowed in code; all must come from `config`.

### 1.2 Outputs
The kernel produces the following outputs via `explain()` or `fit()` return:

1.  **`theory`** (`RuleBase`): A set of logical rules.
    *   Format: Horn Clauses with Negation-as-Failure.
    *   Structure: `Head :- Body1, Body2, ...`
2.  **`metrics`** (`Dict[str, float]`): Performance metrics on the training set.
    *   `precision`, `recall`, `f1_score`, `complexity`, `rule_count`.

### 1.3 Non-Clauses (What the Kernel does NOT do)
*   **No Data Loading:** The kernel does not read files or parse CSVs. It expects `Scene` objects.
*   **No UI/Visualization:** The kernel does not generate graphs or plots. It returns structured data.
*   **No Persistence:** The kernel does not save itself to disk (external serializers do that).

---

## 2. The Sacred API

The following methods constitute the stable public interface of the kernel.

```python
class GDMKernel:
    def fit(self, scenes: List[Scene], target_predicate: str, config: Optional[KernelConfig] = None) -> 'GDMKernel':
        """
        Trains the kernel on the provided scenes.
        Returns self for chaining.
        """
        pass

    def predict(self, scenes: List[Scene]) -> List[bool]:
        """
        Infers the target predicate for a list of scenes.
        Returns a list of booleans (True/False).
        """
        pass

    def explain(self) -> Dict[str, Any]:
        """
        Returns the internal theory in a structured format.
        """
        pass
```

---

## 3. Internal Representation

### 3.1 Terms & Literals
*   **`Literal`**: The atomic unit of logic.
    *   Structure: `predicate(arg1, arg2, ...)`
    *   Example: `father(adam, cain)`
    *   **Negation**: Handled via `negated` boolean flag. `NOT father(x, y)` is `Literal("father", (x, y), negated=True)`.

### 3.2 Rules
*   **`Rule`**: A Horn clause.
    *   Structure: `Head :- Body`
    *   **`Head`**: A single `Literal`.
    *   **`Body`**: A list of `Literal`s (conjunction).
    *   **`RuleID`**: An opaque identifier object (e.g., `RuleID("R_a1b2")`). **Never parse this string for logic.**

### 3.3 Scenes
*   **`Scene`**: A bounded context for learning/inference.
    *   Contains `id`, `facts` (`FactBase`), `target_entity`, `ground_truth`.

---

## 4. Invariants & Safety

1.  **Determinism**: Given the same `config` (with fixed seed) and `scenes`, `fit()` must produce the **exact same** `theory`.
2.  **Termination**: The learning loop must be bounded by `max_depth` and `time_budget`. Infinite loops are forbidden.
3.  **Type Safety**: All internal components must use strict Python type hints. `Any` is prohibited in core logic.
