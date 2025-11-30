import logging
from typing import List, Tuple, Dict, Any
from .types import FailureContext

logger = logging.getLogger(__name__)

class DiscriminativeFeatureSelector:
    """
    CORTEX-OMEGA Optimization: Noise Convergence.
    Selects features that correlate with the target predicate in memory.
    Uses a simplified Information Gain / Correlation metric.
    """
    def __init__(self, min_score: float = 0.01):
        self.min_score = min_score

    def select_features(self, ctx: FailureContext) -> List[str]:
        """
        Returns a list of predicates that are relevant.
        """
        if not ctx.memory:
            return [] # No memory, assume everything is relevant (or nothing?)
            
        # 1. Collect statistics
        # We want to know P(Target | Feature) vs P(Target)
        # Or simpler: Correlation between Feature Presence/Value and Target Truth.
        
        # Structure: feature_key -> {target_true: count, target_false: count}
        # feature_key = (predicate, value) or just predicate?
        # Let's use (predicate, value) for precise correlation.
        
        stats = {} 
        total_pos = 0
        total_neg = 0
        
        # CORTEX-OMEGA v1.4: Confounder Invariance
        # We need to check if feature correlation is stable across time/splits.
        relevant_scenes = [s for s in ctx.memory if s.ground_truth is not None]
        
        if not relevant_scenes:
            return []
            
        for s in relevant_scenes:
            is_positive = s.ground_truth
            if is_positive:
                total_pos += 1
            else:
                total_neg += 1
                
            for pred, args_set in s.facts.facts.items():
                for args in args_set:
                    # Handle unary/binary/n-ary
                    # For unary: args=(entity,) -> val=True (implicitly)
                    # For binary: args=(entity, val) -> val=val
                    if len(args) == 1:
                        val = "true" # Unary predicate presence
                    else:
                        val = str(args[1]) # Value
                        
                    key = (pred, val)
                    if key not in stats:
                        stats[key] = {"pos": 0, "neg": 0}
                        
                    if is_positive:
                        stats[key]["pos"] += 1
                    else:
                        stats[key]["neg"] += 1
                            
        # 2. Calculate Scores
        # Score = |P(Pos|Feature) - P(Pos)|
        # Base probability
        p_pos_base = total_pos / (total_pos + total_neg) if (total_pos + total_neg) > 0 else 0.5
        
        scores = {} # predicate -> max_score (we care if the predicate is useful)
        
        for (pred, val), counts in sorted(stats.items()):
            n_feat = counts["pos"] + counts["neg"]
            if n_feat < 3: continue # Ignore rare features (noise)
            
            p_pos_feat = counts["pos"] / n_feat
            
            # Information Gain-ish: How much does knowing this feature change probability?
            impact = abs(p_pos_feat - p_pos_base)
            
            # Weight by support? A feature that appears once is noisy.
            # Support weight: n_feat / total_samples
            support = n_feat / len(relevant_scenes)
            
            # Final Score
            score = impact * support
            
            # CORTEX-OMEGA v1.5: Feature Priors
            feature_priors = getattr(ctx, "feature_priors", {}) or {}
            if pred in feature_priors:
                score *= feature_priors[pred]
            
            # CORTEX-OMEGA v1.4: Confounder Invariance (Stability Check)
            # Split memory into Early/Late to detect drift.
            # If feature correlation flips or drops, it's likely a confounder.
            if len(relevant_scenes) >= 10:
                mid = len(relevant_scenes) // 2
                early_scenes = relevant_scenes[:mid]
                late_scenes = relevant_scenes[mid:]
                
                def get_local_impact(scenes):
                    l_pos = 0
                    l_feat_pos = 0
                    l_feat_total = 0
                    for s in scenes:
                        if s.ground_truth: l_pos += 1
                        ent = s.target_entity
                        has_feat = False
                        for p, args_set in s.facts.facts.items():
                            if p == pred:
                                for args in args_set:
                                    # Check value match
                                    v = "true"
                                    if len(args) > 1: v = str(args[1])
                                    if v == val:
                                        has_feat = True
                                        break
                            if has_feat: break
                        
                        if has_feat:
                            l_feat_total += 1
                            if s.ground_truth: l_feat_pos += 1
                            
                    if l_feat_total == 0: return 0.0
                    l_base = l_pos / len(scenes) if scenes else 0.5
                    l_prob = l_feat_pos / l_feat_total
                    return abs(l_prob - l_base)

                impact_early = get_local_impact(early_scenes)
                impact_late = get_local_impact(late_scenes)
                
                # Stability penalty: if impact drops significantly, penalize
                stability = 1.0
                if impact_early > 0.1 and impact_late < 0.05:
                    stability = 0.2 # Penalize unstable features
                
                score *= stability
                # logger.debug(f"Feature {pred}={val} Stability={stability:.2f} (Early={impact_early:.2f}, Late={impact_late:.2f})")
            
            if pred not in scores:
                scores[pred] = 0.0
            scores[pred] = max(scores[pred], score)
            
        # 3. Filter
        selected = [pred for pred, score in scores.items() if score > self.min_score]
        
        # Always include relational predicates?
        # selected.extend(["left_of", "behind", "above", "below"])
        
        if selected:
            # Sort by score for debug
            selected.sort(key=lambda p: scores[p], reverse=True)
            logger.debug(f"CORTEX: Feature Selector prioritized {selected[:5]} (Top Score={scores[selected[0]]:.2f}).")
            
        return selected

    def select_numeric_splits(self, ctx: FailureContext) -> List[Tuple[str, str, float, float]]:
        """
        CORTEX-OMEGA v1.4: Numeric Threshold Learning.
        Scans memory for numeric features and finds best splits.
        Returns list of (predicate, operator, threshold, score).
        """
        if not ctx.memory:
            return []
            
        splits = []
        
        # 1. Collect numeric values and labels
        # predicate -> list of (value, is_positive)
        numeric_data = {}
        
        relevant_scenes = [s for s in ctx.memory if s.target_predicate == ctx.target_predicate]
        if not relevant_scenes:
            return []
            
        total_pos = sum(1 for s in relevant_scenes if s.ground_truth)
        total_neg = len(relevant_scenes) - total_pos
        p_pos_base = total_pos / len(relevant_scenes) if len(relevant_scenes) > 0 else 0.5
        
        for scene in relevant_scenes:
            is_positive = scene.ground_truth
            ent = scene.target_entity
            
            for pred, args_set in scene.facts.facts.items():
                for args in args_set:
                    if len(args) == 2 and args[0] == ent:
                        val_str = args[1]
                        try:
                            val = float(val_str)
                            if pred not in numeric_data:
                                numeric_data[pred] = []
                            numeric_data[pred].append((val, is_positive))
                        except ValueError:
                            continue # Not numeric
                            
        # 2. Find best split for each predicate
        for pred, data in numeric_data.items():
            # Sort by value
            data.sort(key=lambda x: x[0])
            # logger.debug(f"Numeric data for {pred}: {data}")
            
            best_score = -1.0
            best_split = None
            if len(data) < 5: continue # Need enough data
            
            data.sort(key=lambda x: x[0])
            
            best_score = 0.0
            best_split = None
            
            # Try all midpoints
            for i in range(len(data) - 1):
                if data[i][0] == data[i+1][0]: continue # Skip duplicate values
                
                threshold = (data[i][0] + data[i+1][0]) / 2.0
                
                # Calculate stats for > Threshold
                # (We can optimize this, but O(N^2) is fine for small memory)
                greater_pos = 0
                greater_total = 0
                
                for val, is_pos in data:
                    if val > threshold:
                        greater_total += 1
                        if is_pos:
                            greater_pos += 1
                            
                if greater_total == 0 or greater_total == len(data): continue
                
                p_pos_gt = greater_pos / greater_total
                impact_gt = abs(p_pos_gt - p_pos_base)
                support_gt = greater_total / len(data)
                score_gt = impact_gt * support_gt
                
                # Calculate stats for <= Threshold (Complement)
                less_pos = total_pos - greater_pos
                less_total = len(data) - greater_total
                p_pos_le = less_pos / less_total
                impact_le = abs(p_pos_le - p_pos_base)
                support_le = less_total / len(data)
                score_le = impact_le * support_le
                
                # Pick the direction that has higher impact
                # logger.debug(f"Split {pred} > {threshold}: score_gt={score_gt:.3f}, score_le={score_le:.3f}")
                
                if score_gt > best_score and score_gt > self.min_score:
                    best_score = score_gt
                    best_split = (pred, ">", threshold, score_gt)
                if score_le > best_score and score_le > self.min_score:
                    best_score = score_le
                    best_split = (pred, "<=", threshold, score_le)
                    
            if best_split:
                splits.append(best_split)
                
        # Sort by score
        splits.sort(key=lambda x: x[3], reverse=True)
        return splits
