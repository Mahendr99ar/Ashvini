# debris-families

Group fragments into break-up families and attribute new pieces.

**Data:** CelesTrak GP debris

**Split:** Clustering on all pieces; attribution trains on 75% of pieces and tests on the rest.

**Baseline:** Nearest mean orbit plane. This model beats it on held-out data.

| Metric | Value |
|---|---|
| clustering.adjusted_rand_index | 0.7844 |
| clustering.clusters_found | 3 |
| clustering.families | 3 |
| clustering.noise_share | 0.1178 |
| attribution.nearest_plane_top1 | 0.6396 |
| attribution.lightgbm_top1 | 0.9911 |

## Limitations

- Families that share an orbit (two break-ups in one plane) cannot be separated by elements alone.
- The designator labels the launch, which merges pieces from separate events on the same launch.
