from cortex_omega.core.rules import parse_rule

try:
    rule_str = "fraud(X) :- transaction(X, amount, V), V > 10000, unverified(X)"
    print(f"Parsing: {rule_str}")
    rule = parse_rule(rule_str)
    print(f"Success: {rule}")
except Exception as e:
    print(f"Failed: {e}")
