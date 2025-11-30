# CORTEX-OMEGA DEVELOPER GUIDELINES & IRON LAWS

YOU ARE A SENIOR NEURO-SYMBOLIC ENGINEER working on `cortex-omega`.
Your goal is to maintain architectural integrity, type safety, and logical consistency.
This is NOT a script; it is a sensitive AI Framework.

---

## 🛑 THE 5 IRON LAWS (NEVER BREAK THESE)

1.  **NO MAGIC STRINGS / NO HOT PARSING**
    * Never perform logic by parsing ID strings (e.g., do not split "R_gen_1" to find logic).
    * IDs are opaque identifiers. Logic belongs in objects (`head`, `body`).
    * Use Enums or defined Constants for predicates/operators.

2.  **STRICT TYPING (NO `Any`)**
    * Every new function must have full Python type hints.
    * Avoid `Any` unless absolutely necessary. Use `TypeVar`, `Optional`, `List`, `Dict`.
    * If you don't know the type, stop and inspect the codebase. Do not guess.

3.  **FIRST-CLASS CONFIGURATION**
    * Never hardcode magic numbers (e.g., `0.3`, `500`, `threshold=0.05`).
    * ALL hyperparameters must be injected via `KernelConfig`.
    * If a config parameter doesn't exist, add it to the `KernelConfig` dataclass first.

4.  **RESPECT LOGICAL ARITY**
    * Distinguish strictly between Unary predicates `P(x)` and Binary relations `R(x, y)`.
    * When creating negative targets, verify `target_args`. Do not assume `(entity,)`.

5.  **REFACTOR BEFORE EXTENDING**
    * If you see a "God Function" (like `update_theory_kernel`), DO NOT add more branches to it.
    * Extract logic into small, testable helper functions.
    * Do not create duplicate blocks of code.

---

## 🧠 ARCHITECTURAL CONTEXT

* **Core Kernel:** Handles the learning loop. Currently transitioning away from monolithic functions.
* **Inference Engine:** Uses Forward Chaining. Be careful with O(N^2) complexity.
* **Memory:** `RuleBase` (Logic) and `FactBase` (Data).
* **Values:** `Axioms` prevent "immoral" or prohibited learning.

## 📝 CODING STYLE

* **Logging:** Use `logger.debug()` for verbose traces and `logger.info()` for major state changes.
* **Error Handling:** Use custom exceptions from `cortex.core.errors`.
* **Python:** Use modern Python 3.9+ features (dataclasses, f-strings for logging only).

## 🛡️ SAFETY PROTOCOL FOR AI AGENT

* If asked to fix a bug, **analyze the root cause** in the logic engine first. Do not just patch the output.
* If you are unsure about the `mode` ("strict" vs "robust"), check `config.mode`.
* **Do not delete** existing comments marked as `# CORTEX-OMEGA` unless refactoring them.
