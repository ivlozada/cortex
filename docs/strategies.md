# Cortex-Omega Repair Strategies

This document describes the strategies used by the `HypothesisGenerator` to repair logical theories.

## Overview

The `HypothesisGenerator` employs a **Strategy Pattern** to propose patches for failing rules. Each strategy focuses on a specific type of logical defect or feature.

## Core Strategies

### 1. NumericThresholdStrategy
**Goal:** Discover numeric split points that distinguish positive from negative examples.
*   **Trigger:** `FALSE_POSITIVE` or `FALSE_NEGATIVE` errors involving numeric features.
*   **Mechanism:**
    1.  Collects numeric values for the target feature from memory.
    2.  Calculates Information Gain for potential split points.
    3.  Proposes conditions like `age(X, V), V > 18`.

### 2. TemporalConstraintStrategy
**Goal:** Identify temporal orderings between events.
*   **Trigger:** Failures in temporal sequences.
*   **Mechanism:**
    1.  Analyzes timestamps of events related to the target.
    2.  Checks for consistent ordering (e.g., `A` always precedes `B`).
    3.  Proposes conditions like `timestamp(A, T1), timestamp(B, T2), T1 < T2`.

### 3. PropertyFilterStrategy (Legacy: `_strategy_add_property_filter`)
**Goal:** Add categorical constraints.
*   **Trigger:** General failures.
*   **Mechanism:**
    1.  Finds properties common to positive examples but missing in negatives (or vice versa).
    2.  Proposes literals like `color(X, red)` or `NOT color(X, blue)`.

### 4. RelationalConstraintStrategy (Legacy: `_strategy_add_relational_constraint`)
**Goal:** Add relational conditions.
*   **Trigger:** Failures involving relationships between entities.
*   **Mechanism:**
    1.  Identifies relations (e.g., `parent(X, Y)`) that are predictive.
    2.  Proposes literals like `parent(X, Y), female(Y)`.

## Extensibility

To add a new strategy:
1.  Inherit from `RepairStrategy` in `cortex_omega.core.strategies`.
2.  Implement the `propose(ctx: FailureContext, features: Dict) -> List[Patch]` method.
3.  Register the strategy in `HypothesisGenerator.__init__`.
