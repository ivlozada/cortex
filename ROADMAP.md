# Cortex Omega Roadmap

## v1.2.0: The "Glass Box" Update
### Core Features
- **Full Epistemic Traceback**: Implement a history stack to track the full derivation path of a conclusion (e.g., `['greek(X)', 'human(X)', 'mortal(X)']`).
- **Debuggable Logic**: Provide detailed failure reports (e.g., "Failed at Step 14 because Rule R392 contradicted Fact F102").
- **Justification API**: Ensure every query result includes a structured `proof` object, not just a confidence score.

### Goal
Transform Cortex-Omega from a "Black Box" probabilistic engine into a verifiable "Glass Box" system that provides receipts for its truths.

## v1.3.0: The "Holographic Logic" Upgrade (Released)
### Core Features
- **Bayesian Rule Scoring**: Robustness to noise via Beta Distribution.
- **Holographic Conflict Resolution**: "Shadow Rules" for handling exceptions without losing generalization.
- **Causal Feature Prioritization**: Anti-confounder logic.

## v1.4.0: The "World-Class" Core Upgrade (Released)
### Strategy
Transform Cortex into a transparent, verifiable engine by exposing internal statistics and enforcing generalization pressure.

### Features
- **First-Class Rule Statistics**: `fires_pos`, `fires_neg`, `reliability`, and `coverage` exposed for every rule.
- **Generalization Pressure (MDL)**: Minimum Description Length scoring to prune redundant rules and favor simplicity.
- **Transparent Auditing**: `inspect_rules` API for deep introspection.

## v1.5.0: The "Killer Vertical" (Released)
### Strategy
Focus on becoming the undisputed best-in-class engine for **Fraud Detection** and **Policy Enforcement** by adding temporal, relational, and strict logic capabilities.

### Features
- **Temporal Sequence Learning**: Learns time-ordered patterns (`A then B`).
- **Relational Logic**: Learns multi-entity relationships (`grandparent(X, Y)`).
- **Strict Mode**: "Zero Tolerance" option for logical purity.
- **Scikit-Learn Wrapper**: `CortexClassifier` for ML pipeline integration.
- **Rule Export**: JSON/Prolog export for audit compliance.

## v1.6.0: The "Neuro-Symbolic Bridge" (LLM Integration)
### Strategy
Position Cortex as the "Logical Guardrail" for LLMs. "No prompts. No hallucinations. Just logic."

### Features
- **LLM-to-Rule Converter**: Utility to parse natural language policy documents into Cortex rules (via LLM API).
- **Hallucination Check**: Verify LLM outputs against the Cortex Fact Store.
- **Hybrid Pipeline**: `LLM(Unstructured) -> Cortex(Structured) -> Action`.

