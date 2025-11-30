from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from .types import FailureContext, Patch, PatchOperation
from .config import KernelConfig
from .selectors import DiscriminativeFeatureSelector
from .inference import InferenceEngine
from .rules import RuleBase

class RepairStrategy(ABC):
    """
    Abstract Base Class for Repair Strategies.
    Each strategy proposes patches to fix a specific type of logical failure.
    """
    def __init__(self, config: KernelConfig):
        self.config = config

    @abstractmethod
    def propose(self, ctx: FailureContext, features: Dict[str, Any]) -> List[Patch]:
        """
        Proposes a list of patches given the failure context and extracted features.
        """
        pass

    @property
    def name(self) -> str:
        return self.__class__.__name__

class NumericThresholdStrategy(RepairStrategy):
    """
    CORTEX-OMEGA v1.4: Numeric Threshold Learning.
    Uses DiscriminativeFeatureSelector.select_numeric_splits to propose
    numeric constraints like:

        attr(X, V_attr), V > 3.5

    as additional literals in the rule body.
    """

    def __init__(self, config: KernelConfig):
        super().__init__(config)
        self.selector = DiscriminativeFeatureSelector(min_score=0.02)

    def propose(self, ctx: FailureContext, features: Dict[str, Any]) -> List[Patch]:
        patches: List[Patch] = []

        # We only try this if we have memory; otherwise numeric stats are meaningless
        if not ctx.memory:
            return patches

        # Ask selector for best numeric splits per predicate
        splits = self.selector.select_numeric_splits(ctx)
        if not splits:
            return patches

        # We’ll just take the top few (3) to avoid explosion
        for pred, op, threshold, score in splits[:3]:
            # Canonical pattern:
            #   attr(X, V_attr), V_attr > threshold   OR   V_attr <= threshold
            # We implement this via ADD_LITERAL with "add_body" so PatchApplier
            # uses parse_literal on each string.
            var_name = f"V_{pred}"

            add_body = [
                f"{pred}(X, {var_name})",       # bind numeric value to var_name
                f"{var_name} {op} {threshold}", # numeric comparison literal
            ]

            patch = Patch(
                operation=PatchOperation.ADD_LITERAL,
                target_rule_id=str(ctx.rule.id),
                details={
                    "add_body": add_body,
                    "split_predicate": pred,
                    "operator": op,
                    "threshold": threshold,
                },
                confidence=float(score),
                explanation=(
                    f"Numeric split on {pred}: prefer cases where "
                    f"{var_name} {op} {threshold:.4f}"
                ),
                source_strategy=self.name
            )
            patches.append(patch)

        return patches

class TemporalConstraintStrategy(RepairStrategy):
    """
    CORTEX-OMEGA v1.4: Temporal Sequence Learning.
    If a rule fires on a negative example (False Positive), try to add a temporal constraint
    (T2 > T1) that holds for positives but fails for this negative.
    """
    def propose(self, ctx: FailureContext, features: Dict[str, Any]) -> List[Patch]:
        patches = []
        if ctx.error_type != "FALSE_POSITIVE":
            return []
            
        # 1. Find all variables in the rule body
        variables = set()
        for lit in ctx.rule.body:
            for arg in lit.args:
                if isinstance(arg, str) and arg and arg[0].isupper():
                    variables.add(arg)
                    
        if len(variables) < 2:
            return []
            
        # 2. Get bindings for the current negative firing
        engine = InferenceEngine(ctx.scene_facts, RuleBase())
        bindings_list = engine.evaluate_body(ctx.rule.body)
        
        if not bindings_list:
            return []
            
        # Check each grounding (usually just one for specific counter-example)
        for bindings in bindings_list:
            # Identify numeric values (timestamps)
            numeric_vars = []
            for var, val in bindings.items():
                try:
                    # Handle Term objects: if it's a Term, we can't convert to float unless it's a simple constant
                    if hasattr(val, 'name') and not val.args:
                        val_str = val.name
                    elif isinstance(val, str):
                        val_str = val
                    else:
                        continue # Skip complex terms for numeric/temporal strategies
                        
                    float(val_str)
                    numeric_vars.append((var, float(val_str)))
                except ValueError:
                    continue
            
            if len(numeric_vars) < 2:
                continue
                
            # Try all pairs
            for v1, val1 in numeric_vars:
                for v2, val2 in numeric_vars:
                    if v1 == v2: continue
                    
                    # We want a constraint that FAILS here (to kill the FP).
                    # So if we propose v1 > v2, it must be that val1 <= val2 currently.
                    if val1 <= val2:
                        # Potential constraint: v1 > v2
                        # Check if this holds for POSITIVES in memory.
                        
                        consistent_with_positives = True
                        if ctx.memory:
                            for s in ctx.memory:
                                if not s.ground_truth: continue
                                if s.target_predicate != ctx.target_predicate: continue
                                
                                # Find bindings for s
                                matcher_pos = InferenceEngine(s.facts, RuleBase())
                                pos_bindings_list = matcher_pos.evaluate_body(ctx.rule.body)
                                if not pos_bindings_list: continue 
                                
                                # Check if v1 > v2 holds for at least one grounding in this positive
                                satisfied_in_s = False

                                for pb in pos_bindings_list:
                                    try:
                                        pval1 = float(pb.get(v1, "0"))
                                        pval2 = float(pb.get(v2, "0"))
                                        if pval1 > pval2:
                                            satisfied_in_s = True
                                            break
                                    except:
                                        pass
                                
                                if not satisfied_in_s:
                                    consistent_with_positives = False
                                    break
                        
                        if consistent_with_positives:
                            # Found a valid temporal constraint!
                            patch = Patch(
                                operation=PatchOperation.ADD_LITERAL,
                                target_rule_id=str(ctx.rule.id),
                                details={
                                    "add_body": [f">({v1}, {v2})"]
                                },
                                confidence=self.config.hyperparams.temporal_confidence_threshold,
                                source_strategy=self.name
                            )
                            patches.append(patch)
                            
        return patches

class RecursiveStructureStrategy(RepairStrategy):
    """
    CORTEX-OMEGA v2.0: Recursive Structure Learning.
    Detects patterns like p(s(X)) :- p(X).
    
    It looks for relationships between the arguments of the current failure
    and the arguments of other positive examples in memory.
    """
    def propose(self, ctx: FailureContext, features: Dict[str, Any]) -> List[Patch]:
        patches = []
        # Only relevant for False Negatives (we missed a positive case)
        if ctx.error_type != "FALSE_NEGATIVE":
            return []
            
        if not ctx.target_args:
            return []
            
        # 1. Analyze target arguments of the current failure
        # e.g., add(s(0), s(0), s(s(0)))
        # We want to see if we can peel 's' off any argument and find a matching positive in memory.
        
        current_args = ctx.target_args # Tuple of strings or Terms
        
        # We need to parse them into Terms if they are strings
        from .rules import parse_term, Term, Rule, RuleID, Literal
        
        parsed_args = []
        for arg in current_args:
            if isinstance(arg, str):
                parsed_args.append(parse_term(arg))
            else:
                parsed_args.append(arg)
                
        # 2. Try peeling each argument
        # We look for a "predecessor" state in memory.
        # Predecessor means: same predicate, but arguments are "smaller".
        
        if not ctx.memory:
            return []
            
        # Build index of positive examples in memory
        memory_positives = []
        for s in ctx.memory:
            if s.ground_truth and s.target_predicate == ctx.target_predicate:
                s_args = []
                for a in s.target_args:
                    if isinstance(a, str): s_args.append(parse_term(a))
                    else: s_args.append(a)
                memory_positives.append(tuple(s_args))
                
        # 3. Search for recursive pattern
        # Pattern: p(A, s(B), s(C)) :- p(A, B, C)
        # We check if peeling one or more args of 'current_args' leads to a tuple in 'memory_positives'.
        
        # Generalized Search: Peel ANY combination of args
        # For N args, there are 2^N - 1 combinations. N is small (3 for add).
        import itertools
        indices = range(len(parsed_args))
        
        for r in range(1, len(parsed_args) + 1):
            for subset_indices in itertools.combinations(indices, r):
                # Check if all selected args are peelable (have same wrapper, e.g. 's')
                wrapper = None
                peelable = True
                for idx in subset_indices:
                    arg = parsed_args[idx]
                    if not isinstance(arg, Term) or not arg.args or len(arg.args) != 1:
                        peelable = False
                        break
                    if wrapper is None:
                        wrapper = arg.name
                    elif arg.name != wrapper:
                        peelable = False # Mixed wrappers, maybe too complex for now
                        break
                
                if not peelable:
                    continue
                    
                # Construct predecessor tuple
                pred_args = list(parsed_args)
                vars_map = {} # Map index to variable name
                
                for idx in range(len(parsed_args)):
                    if idx in subset_indices:
                        # Peeled
                        pred_args[idx] = parsed_args[idx].args[0]
                        vars_map[idx] = f"V_{idx}" # The inner value
                    else:
                        # Kept as is
                        vars_map[idx] = f"K_{idx}" # The constant value
                        
                if tuple(pred_args) in memory_positives:
                    # Found recursive support!
                    # Construct the rule:
                    # head: p(K_0, s(V_1), s(V_2)) 
                    # body: p(K_0, V_1, V_2)
                    
                    head_args = []
                    body_args = []
                    
                    for idx in range(len(parsed_args)):
                        if idx in subset_indices:
                            # Head has s(V)
                            head_args.append(Term(wrapper, (Term(vars_map[idx]),)))
                            body_args.append(Term(vars_map[idx]))
                        else:
                            # Head has K (constant variable)
                            head_args.append(Term(vars_map[idx]))
                            body_args.append(Term(vars_map[idx]))
                            
                    # Create Rule
                    # head: target_pred(...)
                    # body: target_pred(...)
                    
                    # Head Literal
                    h_args_terms = tuple(head_args) # These are Terms with vars
                    h_lit = Literal(ctx.target_predicate, h_args_terms)
                    
                    # Body Literal
                    b_args_terms = tuple(body_args)
                    b_lit = Literal(ctx.target_predicate, b_args_terms)
                    
                    new_rule = Rule(
                        id=RuleID("recursive_candidate"),
                        head=h_lit,
                        body=[b_lit]
                    )
                    
                    patch = Patch(
                        operation=PatchOperation.CREATE_BRANCH,
                        target_rule_id=str(ctx.rule.id), # Not used for branch
                        details={
                            "rule": new_rule
                        },
                        confidence=0.9, # High confidence for recursion
                        explanation=f"Recursive pattern found: {h_lit} :- {b_lit}",
                        source_strategy=self.name
                    )
                    patches.append(patch)
                    
        return patches

class ArgumentGeneralizationStrategy(RepairStrategy):
    """
    CORTEX-OMEGA v2.0: Argument Generalization.
    Detects patterns in the arguments of the target predicate itself.
    e.g. add(X, 0, X) -> 2nd arg is constant 0, 1st and 3rd are identical.
    """
    def propose(self, ctx: FailureContext, features: Dict[str, Any]) -> List[Patch]:
        patches = []
        if ctx.error_type != "FALSE_NEGATIVE":
            return []
            
        if not ctx.memory:
            return []
            
        # Collect all positive examples for this predicate
        positives = [s.target_args for s in ctx.memory 
                     if s.ground_truth and s.target_predicate == ctx.target_predicate and s.target_args]
                     
        if not positives:
            return []
            
        # Check for consistency across ALL positives (or a significant subset)
        # But wait, we might have multiple rules (base case AND recursive case).
        # So we shouldn't expect ALL positives to match.
        # We should look for a cluster that includes the CURRENT failure.
        
        current_args = ctx.target_args
        # Add current to the set to analyze
        cluster = positives + [current_args]
        
        # We need to find a pattern that covers 'current_args' and some 'positives'.
        # Let's try to generalize 'current_args' against each positive in memory.
        
        from .rules import Term, Rule, RuleID, Literal, parse_term
        
        def terms_equal(t1, t2):
            return str(t1) == str(t2)
            
        for pos_args in positives:
            if len(pos_args) != len(current_args): continue
            
            # Anti-unify current_args and pos_args
            # Result is a list of generalized args and a set of constraints
            
            gen_args = []
            constraints = {} # var_name -> val (not used for base case generation usually)
            
            # We want to detect:
            # 1. Constants: arg[i] is same in both
            # 2. Identity: arg[i] == arg[j] in both
            
            # Let's build a template
            # Start with all variables
            template = [f"V{i}" for i in range(len(current_args))]
            
            # 1. Check for Constants
            for i in range(len(current_args)):
                if terms_equal(current_args[i], pos_args[i]):
                    # It's a constant (at least between these two)
                    # We can use the term itself
                    template[i] = current_args[i]
            
            # 2. Check for Identity (Variable sharing)
            # If template[i] is still a variable, check if it matches template[j]
            # But we need to check if the underlying values match in BOTH examples
            
            # Group indices by value in current_args
            current_groups = {}
            for i, arg in enumerate(current_args):
                s = str(arg)
                if s not in current_groups: current_groups[s] = []
                current_groups[s].append(i)
                
            # Group indices by value in pos_args
            pos_groups = {}
            for i, arg in enumerate(pos_args):
                s = str(arg)
                if s not in pos_groups: pos_groups[s] = []
                pos_groups[s].append(i)
                
            # Intersect groups
            # If indices {0, 2} have same value in current AND same value in pos,
            # then we can share the variable.
            
            # Map index to variable name
            var_map = {}
            next_var = 0
            
            for i in range(len(current_args)):
                if not isinstance(template[i], str) or not template[i].startswith("V"):
                    continue # Already a constant
                    
                if i in var_map:
                    template[i] = var_map[i]
                    continue
                    
                # Start a new variable group
                var_name = f"X{next_var}"
                next_var += 1
                var_map[i] = var_name
                template[i] = var_name
                
                # Check other indices
                for j in range(i+1, len(current_args)):
                    # Check if i and j share value in current AND pos
                    if terms_equal(current_args[i], current_args[j]) and \
                       terms_equal(pos_args[i], pos_args[j]):
                        var_map[j] = var_name
                        
            # Now we have a candidate rule head: target(template)
            # e.g. add(X0, 0, X0)
            
            # Convert template to Terms/Strings
            final_args = []
            for item in template:
                if isinstance(item, str) and item.startswith("X"):
                    final_args.append(item) # Variable
                elif isinstance(item, str) and item.startswith("V"):
                     # Unique variable (no sharing found)
                     final_args.append(item.replace("V", "Y"))
                else:
                    final_args.append(item) # Constant Term
            
            # Create Rule
            # head: target(final_args)
            # body: [] (Fact / Base case)
            
            # We need to convert final_args to tuple of Terms/strings
            # Literal expects tuple
            
            head_lit = Literal(ctx.target_predicate, tuple(final_args))
            rule_id_str = f"base_case_{hash(str(head_lit))}"
            new_rule = Rule(RuleID(rule_id_str), head_lit, [])
            
            patch = Patch(
                operation=PatchOperation.CREATE_BRANCH,
                target_rule_id=str(ctx.rule.id),
                details={
                    "rule": new_rule
                },
                confidence=0.95,
                explanation=f"Base case pattern found: {head_lit}",
                source_strategy=self.name
            )
            patches.append(patch)
            
        return patches
