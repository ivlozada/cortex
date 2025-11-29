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

## v1.4.0: The "Killer Vertical" (Fraud & Policy)
### Strategy
Focus on becoming the undisputed best-in-class engine for **Fraud Detection** and **Policy Enforcement**.

### Features
- **Domain Helpers**: `cortex_omega.fraud` module with pre-built rules for velocity, geolocation, and risk scoring.
- **Pandas Integration**: Native `absorb_dataframe(df)` method for high-throughput ingestion.
- **Rule Export**: Standardized JSON/YAML export for audit compliance.

## v1.5.0: The "Neuro-Symbolic Bridge" (LLM Integration)
### Strategy
Position Cortex as the "Logical Guardrail" for LLMs. "No prompts. No hallucinations. Just logic."

### Features
- **LLM-to-Rule Converter**: Utility to parse natural language policy documents into Cortex rules (via LLM API).
- **Hallucination Check**: Verify LLM outputs against the Cortex Fact Store.
- **Hybrid Pipeline**: `LLM(Unstructured) -> Cortex(Structured) -> Action`.

