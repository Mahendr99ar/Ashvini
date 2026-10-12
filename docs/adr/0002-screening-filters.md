# 2. Three-stage screening

**Status:** accepted, 2026-10-10

## Decision

1. Apogee/perigee filter (Hoots et al., 1984) on altitude bands, with the report distance plus 10 km as padding.
2. A 30-second grid of positions and a k-d tree per step, with a search radius of the report distance plus the distance two objects can close in half a step (15.8 km/s head-on).
3. Brent's method on the miss distance within one step either side of the closest sample in each pass.

## Why not a finer grid alone

A 1-second grid over three days for 15,000 objects is 3.9 billion positions. The coarse grid plus refinement finds the same minimum, because along one pass the miss distance has a single minimum and the true one sits within a step of the closest sample.

## Tested by

`tests/test_screening.py` plants encounters with known miss distances and times; screening must find each one once, within 2 seconds and 10 metres.
