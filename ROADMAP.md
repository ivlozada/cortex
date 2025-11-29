# Cortex Omega Roadmap

## v1.2.0: The "Glass Box" Update
### Core Features
- **Full Epistemic Traceback**: Implement a history stack to track the full derivation path of a conclusion (e.g., `['greek(X)', 'human(X)', 'mortal(X)']`).
- **Debuggable Logic**: Provide detailed failure reports (e.g., "Failed at Step 14 because Rule R392 contradicted Fact F102").
- **Justification API**: Ensure every query result includes a structured `proof` object, not just a confidence score.

### Goal
Transform Cortex-Omega from a "Black Box" probabilistic engine into a verifiable "Glass Box" system that provides receipts for its truths.
