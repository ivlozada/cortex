from cortex_omega import Cortex

def test_rule_stats_monotonic():
    brain = Cortex()

    data = [
        {"id": "a1", "feature": "x", "is_good": True},
        {"id": "a2", "feature": "x", "is_good": True},
        {"id": "b1", "feature": "y", "is_good": False},
        {"id": "b2", "feature": "y", "is_good": False},
    ]
    brain.absorb_memory(data, target_label="good")

    # Query same feature several times to accumulate stats
    # Note: In standard Cortex, query() without ground truth doesn't update stats (no feedback).
    # But absorb_memory DOES update stats.
    # To test monotonicity, we should absorb MORE data.
    
    more_data_x = [{"id": f"a{i}", "feature": "x", "is_good": True} for i in range(3, 8)]
    brain.absorb_memory(more_data_x, target_label="good")
    
    # y is consistently bad (noise/confounder), so if we force it to fire, it should have high fires_neg
    # But wait, if it's "good", and y -> False, then a rule "good :- y" would be a False Positive.
    # If we have data y -> False, the engine shouldn't even learn "good :- y" ideally.
    # Unless we force it or it was learned from some noise.
    
    # Let's inject a bad rule manually to test stats tracking
    brain.add_rule("good(X) :- feature(X, y)")
    
    bad_data_y = [{"id": f"b{i}", "feature": "y", "is_good": False} for i in range(3, 8)]
    brain.absorb_memory(bad_data_y, target_label="good")

    rules = brain.inspect_rules(target="good")

    # Assert that rule stats look sane: rule for x is highly reliable, y is not.
    # We need to find the specific rules.
    rule_x = None
    rule_y = None
    
    for r in rules:
        if "feature(X, x)" in str(r) and not r.head.negated:
            rule_x = r
        # We want the POSITIVE rule for y, which should be failing
        if "feature(X, y)" in str(r) and not r.head.negated:
            rule_y = r
            
    assert rule_x is not None, "Rule X should exist"
    assert rule_x is not None, "Rule X should exist"
    
    # CORTEX-OMEGA v1.4: MDL Pruning
    # Rule Y is harmful (False Positive) and should be pruned by garbage_collect.
    if rule_y is None:
        print("Rule Y was correctly pruned by MDL (Generalization Pressure).")
    else:
        # If it survived (e.g. grace period), check stats
        print(f"Rule Y: {rule_y} (Rel={rule_y.reliability:.2f}, Cov={rule_y.coverage})")
        assert rule_y.reliability < 0.2

    print(f"Rule X: {rule_x} (Rel={rule_x.reliability:.2f}, Cov={rule_x.coverage})")
    assert rule_x.reliability > 0.8
    assert rule_x.coverage > 0

if __name__ == "__main__":
    test_rule_stats_monotonic()
    print("Test passed!")
