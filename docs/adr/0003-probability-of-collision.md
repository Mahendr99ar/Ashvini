# 3. Pc with the short-encounter model

**Status:** accepted, 2026-10-10

## Decision

Foster and Estes (1992): combined covariance projected onto the plane perpendicular to the relative velocity, integrated over a disc of the combined hard-body radius (20 m by default) with polar Gauss-Legendre quadrature. The maximum Pc over covariance size (Alfano, 2005) is stored alongside, as a worst case.

## Limits

The model assumes straight-line relative motion, which fails for slow encounters such as co-orbiting objects in the same shell. Those show low relative speed on the event card, and the recommendation does not lean on Pc alone for them.
