"""
Example 02: Relational Learning (Grandparent)
=============================================
Demonstrates Cortex-Omega's ability to learn multi-hop relational rules.
Teaches the concept of "grandparent" from "parent" relations.

Scenario:
- A is parent of B.
- B is parent of C.
- Therefore, A is grandparent of C.
"""

from cortex_omega.api.client import Cortex
from cortex_omega.core.errors import EpistemicVoidError

def main():
    # 1. Initialize
    cortex = Cortex()
    print("🧠 Cortex Initialized.")

    # 2. Teach
    # We define a family tree.
    # Abe -> Homer -> Bart
    # Abe -> Homer -> Lisa
    # Mona -> Homer -> Bart
    
    training_data = [
        # Facts
        {"id": "fact1", "parent": "homer", "child": "bart", "is_grandparent": False},
        {"id": "fact2", "parent": "abe", "child": "homer", "is_grandparent": False},
        
        # Positive Examples (Grandparents)
        # We need to express the relation. 
        # In Cortex, we usually learn properties of an entity.
        # But here we want to learn a relation: grandparent(X, Y).
        # Currently, Cortex's high-level API focuses on "target(Entity)".
        # To learn relations, we model the *pair* or use the lower-level kernel.
        # However, for this example, let's define "is_grandparent" as a property of the *ancestor* 
        # relative to a specific *descendant* context, OR we simply define the facts and ask if X is a grandparent.
        
        # Let's try the standard approach: 
        # We want to learn: grandparent(X, Y) :- parent(X, Z), parent(Z, Y).
        # The input format for relations is:
        # {"head": ("grandparent", "abe", "bart"), "body": [("parent", "abe", "homer"), ("parent", "homer", "bart")]}
        # But the high-level `absorb` expects dictionaries of properties for an ID.
        
        # Let's simplify: We want to classify if X is a grandparent based on graph structure.
        # This requires the "Grandparent" relation to be the target.
        
        # We will feed facts about the world first.
        # Cortex.absorb() is designed for "Entity Classification".
        # To do "Link Prediction" (Relation Learning), we might need to ingest raw facts.
        
        # Workaround for high-level API:
        # We define the "world" with facts, then give examples of the target relation.
    ]
    
    # Actually, let's use the explicit relational ingestor format if available, 
    # or just use `absorb` to build the graph.
    
    # Let's define the family members and their relationships.
    family_data = [
        {"id": "abe", "parent_of": "homer"},
        {"id": "mona", "parent_of": "homer"},
        {"id": "homer", "parent_of": "bart"},
        {"id": "homer", "parent_of": "lisa"},
        {"id": "marge", "parent_of": "bart"},
        {"id": "marge", "parent_of": "lisa"},
    ]
    
    # We want to learn `grandparent_of(X, Y)`.
    # This is a binary predicate. The current `absorb` assumes unary target `target(X)`.
    # However, we can learn `is_grandparent(X)` if `exists Z: parent(X, Z) & parent(Z, Y)`.
    # But that leaves Y unbound.
    
    # Let's stick to the "Grandparent" example from the tests, which uses the Kernel directly.
    # But here we want to use the Client.
    
    # If the Client doesn't support n-ary target learning easily yet, 
    # we can demonstrate a simpler relational concept: "Ancestor" or "Recursive" property?
    # Or we can just use the "Entity" approach:
    # "A person is a grandparent if they have a child who has a child."
    
    training_data = [
        # Abe is a grandparent (has child Homer, who has child Bart)
        {"id": "abe", "child": "homer", "grandchild": "bart", "is_grandparent": True},
        # Homer is a parent but not a grandparent (in this closed world)
        {"id": "homer", "child": "bart", "is_grandparent": False},
        # Bart is a child
        {"id": "bart", "is_grandparent": False},
    ]
    
    # Wait, this flattens the relation.
    # To truly demo relational learning, we need to provide the graph.
    # Let's manually inject facts into the cortex.facts base for the background knowledge.
    
    # 1. Background Knowledge (The Family Tree)
    cortex.facts.add("parent", ("abe", "homer"))
    cortex.facts.add("parent", ("homer", "bart"))
    cortex.facts.add("parent", ("homer", "lisa"))
    
    # 2. Training Examples
    # We want to learn: is_grandparent(X) :- parent(X, Z), parent(Z, Y).
    # Note: The variable Y is existential. "Is grandparent of SOMEONE".
    
    examples = [
        {"id": "abe", "is_grandparent": True},
        {"id": "homer", "is_grandparent": False}, # Has children, but they don't have children
        {"id": "bart", "is_grandparent": False},
    ]
    
    print("📚 Absorbing family tree and examples...")
    cortex.absorb_memory(examples, target_label="is_grandparent")
    
    # 3. Query
    print("\n🔮 Predictions:")
    # Let's add a new branch to test generalization
    # Ned -> Rod -> Todd (Wait, Rod is child of Ned. Todd is child of Ned. No grandchildren.)
    # Let's add: Burns -> Smithers -> (No one). Burns is not grandparent.
    # Let's add: Grampa_Smurf -> Papa_Smurf -> Smurfette.
    
    cortex.facts.add("parent", ("grampa_smurf", "papa_smurf"))
    cortex.facts.add("parent", ("papa_smurf", "smurfette"))
    
    try:
        result = cortex.query(id="grampa_smurf", target="is_grandparent")
        print(f"  Grampa Smurf: {result.prediction}")
        if result.explanation:
            print(f"    Reason: {result.explanation}")
    except EpistemicVoidError:
        print("  Grampa Smurf: Unknown (Epistemic Void)")

    print("\n📜 Learned Rules:")
    for rule in cortex.export_rules():
        print(f"  {rule}")

if __name__ == "__main__":
    main()
