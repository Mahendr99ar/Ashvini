# 4. Every model earns its place against a baseline

**Status:** accepted, 2026-10-10

## Decision

Each module in `ml/` writes a model card with its baseline, the held-out result for both, and a plain yes or no for whether it won. The API serves the collision-risk model only when a card from a real training run says it beat the carried-forward baseline. Recommendations are always shown with the thresholds that produced them, and a person records the decision, with a note required for a manoeuvre or a disagreement.

## Why

In the ESA competition the carried-forward baseline was hard to beat. A model that looks clever and loses to it would make decisions worse.
