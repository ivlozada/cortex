# Design Doc: Cortex-Omega v1.2.0 (The "Glass Box" Update)

## 1. Problem Statement
Currently, Cortex-Omega provides the *final rule* that triggered a conclusion (e.g., `mortal(X) :- human(X)`), but it does not provide the **full derivation path** (the "Traceback").
To truly claim the title of an "Epistemic Engine," Cortex must provide the full justification for its knowledge, distinguishing it from probabilistic "black box" models.

## 2. Proposed Solution: Epistemic Traceback
We will implement a `Traceback` system that records the chain of logical steps taken to reach a conclusion.

### 2.1. The `Proof` Object
Instead of a simple string explanation, the `InferenceResult` will return a structured `Proof` object.

```python
@dataclass
class ProofStep:
    rule_id: str
    rule_str: str
    bindings: Dict[str, str]
    derived_fact: str

@dataclass
class Proof:
    steps: List[ProofStep]
    confidence: float
    
    def __repr__(self):
        # Format: "greek(socrates) -> human(socrates) -> mortal(socrates)"
        return " -> ".join([s.derived_fact for s in self.steps])
```

### 2.2. Inference Engine Update
The `InferenceEngine.forward_chain` method currently stores a flat `trace`. We need to enhance this to reconstruct the causal chain for a specific target.

- **Dependency Graph**: When a fact is derived, we must store *which* rule and *which* antecedent facts caused it.
- **Backtracking**: When a query is made, we traverse this graph backwards from the target fact to the axioms (ground truths).

### 2.3. API Changes
- `InferenceResult.explanation` will be deprecated in favor of `InferenceResult.proof`.
- `brain.query()` will return this rich object.

## 3. User Experience (The "Receipt")

```python
result = brain.query(id="socrates", target="mortal")
print(result.proof)
# Output:
# 1. greek(socrates) [Fact]
# 2. human(socrates) [Rule R1: human(X) :- greek(X)]
# 3. mortal(socrates) [Rule R2: mortal(X) :- human(X)]
```

## 4. Implementation Phases
1. **Graph Storage**: Update `FactBase` or `InferenceEngine` to store provenance metadata for every fact.
2. **Backtracker**: Implement the recursive traceback algorithm.
3. **API Exposure**: Update `InferenceResult` and `client.py`.
4. **Visualization**: (Optional) Generate a Mermaid diagram of the proof.
