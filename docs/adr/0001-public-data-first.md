# 1. Public data first, operator data when offered

**Status:** accepted, 2026-10-10

## Context

Operators decide on manoeuvres using their own ephemerides and conjunction messages from Space-Track or commercial providers. A public tool has none of that. It has CelesTrak's general perturbations data, whose accuracy is of order a kilometre at epoch and grows by about a kilometre per day along track.

## Decision

Screen on CelesTrak data by default, model TLE uncertainty with an age-based covariance (Flohrer et al., 2008 ranges), and say on every event that it is not an official assessment. Accept CCSDS OEM ephemerides and CDMs from operators; their data replaces public data for their own objects.

## Consequences

Pc values from public data are rough. The risk model is trained on ESA's real CDMs, but served on features computed from public data, and its card says that this is a weaker setting than the one it was trained in.
