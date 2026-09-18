# Decision: Tighten evaluation and separate evidence causality from explanation faithfulness

Date: 2026-09-18
Status: Accepted

## Context

Initial documentation had the correct research direction but left several protocol choices underspecified: cross-generator split construction, threshold selection, repeated-run uncertainty, evidence terminology, and the boundary between prediction-level evidence dependence and explanation faithfulness.

## Evidence

Internal review found that these ambiguities could allow test leakage, unstable comparisons, over-claiming attribution maps as evidence, or evaluating explanation faithfulness before proving that prediction depends on the claimed evidence.

## Decision

- Use deterministic generator-disjoint cross-generator splits.
- Select threshold on validation only; AUROC is primary detection metric.
- Target 3 seeds and report mean ± std for stochastic trained baselines when compute permits.
- Distinguish candidate evidence, validated predictive evidence, and explanation-grounded evidence.
- Split the roadmap into R6 Evidence Causality Diagnostics and R8 Explanation Faithfulness, with Stage 3 training in between.
- Strengthen R0 gates to include dataset bias/leakage validity checks.

## Why

These changes improve research validity without adding model complexity. They make claims falsifiable and keep implementation simple.

## Consequence

`README.md`, `AGENTS.md`, research problem, architecture, evaluation protocol, roadmap, and R0 plan must stay synchronized with this decision.
