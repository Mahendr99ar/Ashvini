# Ashvini architecture

Ashvini watches Earth orbit for three kinds of trouble: two objects about to meet, a satellite whose health or orbit is changing, and space weather that is about to make every low orbit less predictable. It is built in three layers, and each one is meant to be shown to a different reader.

- **ML layer:** six models, each trained on a public dataset and each compared with a plain baseline before it is allowed near the product.
- **Product layer:** an operator console with a decision at its centre (manoeuvre, monitor, or ignore), plus the documents in `product/`.
- **Deployment layer (FDE):** the parts that let Ashvini be dropped into an operator's own setup, such as their file formats, their alert channels, their thresholds and their servers.

---

## Data

| Source | What | Refresh | Licence notes |
|---|---|---|---|
| CelesTrak GP (OMM / TLE) | Orbits of active satellites, debris, rocket bodies | Every 4 h by cron, no faster | Free; please keep request rates low |
| Space-Track (free account) | Historical TLEs, decay history, public CDM summaries | Training only | Redistribution restricted. Only derived results leave the machine |
| ESA Collision Avoidance Challenge | Real conjunction data messages from ESA operations | Static | Zenodo 4463683 |
| NOAA SWPC | Real-time solar wind, Kp | Hourly | Public |
| NASA OMNI | Hourly solar wind since 1963 | Training | Public |
| GFZ Potsdam | Definitive Kp | Training | CC BY 4.0 |
| CHAMP / GRACE / Swarm densities | Accelerometer-derived thermosphere density | Training | Public, TU Delft |
| ESA-ADB | Real satellite telemetry with labelled anomalies | Static | Public benchmark |
| SPEED+ | Spacecraft images with ground-truth pose | Static | Stanford / ESA |
| NASA SDO | Solar images | Training | Public |
| JPL CNEOS | Asteroid close approaches | Daily | Public |

---

## Pipeline (Stage 1, built)

```
CelesTrak GP  ->  ingest (cache, rate limit, dedupe by NORAD id + epoch)
              ->  catalogue table (one row per object, latest elements)
              ->  propagate 7 days with SGP4 (vectorised SatrecArray)
              ->  screening
                    1. apogee/perigee filter (Hoots et al. 1984)
                    2. coarse time steps + k-d tree on positions
                    3. TCA refinement with Brent's method on the miss distance
              ->  probability of collision (Foster 2-D, covariance from element age)
              ->  conjunction events (idempotent ids: hash of pair + TCA rounded)
              ->  FastAPI  ->  web console, alert channels, CCSDS exports
```

Rules carried over from GhostWake: the pipeline is idempotent, every alert keeps its lineage (which run, which elements, which thresholds), and analyst decisions live in an append-only review log that a rerun never overwrites.

---

## ML modules

Each module lives in `ml/<module>/` with a training script, a Kaggle or Colab notebook, a model card, and an `eval.json` that the website reads. Nothing ships unless it beats its baseline on held-out data, and the results table says so either way.

| # | Module | Data | Baseline | Models | Metric |
|---|---|---|---|---|---|
| 1 | Collision risk | ESA CDMs | Latest CDM risk carried forward | LightGBM, sequence Transformer over CDMs, deep ensemble, conformal intervals | Challenge score (MSE on high-risk, F2) |
| 2 | Orbit error correction | Space-Track TLE history | Plain SGP4 | Gradient boosting and MLP on SGP4 residuals | Position error at 1, 3, 7 days |
| 3 | Kp forecast | OMNI + GFZ Kp | Persistence, 27-day recurrence | LightGBM, TCN, Chronos and TimesFM (zero-shot, fine-tuned) | RMSE, storm true skill statistic |
| 4 | Thermosphere density | CHAMP, GRACE, Swarm | NRLMSISE-00 | MLP with Monte Carlo dropout, following Licata and Mehta | Log-density error, calibration |
| 5 | Telemetry anomalies | ESA-ADB | Isolation Forest, thresholds | LSTM / Transformer autoencoder, TCN, foundation model | Event-wise F-score from the benchmark |
| 6 | Manoeuvre detection | TLE history, public ISS reboosts, injected events | Mean-motion jump threshold | Change-point detection, gradient boosting on element deltas | Precision, recall, detection delay |
| 7 | Debris families | Catalogue, labels from international designators | Nearest parent by orbit plane | HDBSCAN on elements, classifier, UMAP view | Adjusted Rand index, top-1 parent |
| 8 | Avoidance policy (RL) | Ashvini's own propagator and screening as the environment | Fixed Pc threshold rule | PPO | Collisions avoided per kg of propellant |
| 9 | Spacecraft pose | SPEED+ | Published keypoint baseline | HRNet keypoints + PnP, domain adaptation | ESA pose score |
| 10 | Solar flares | SDO | Climatology | Small CNN / ViT, fine-tuned Surya | True skill statistic |

Modules 1 to 5 are the core. 6 and 7 come next. 8 to 10 are stretch goals and say so on the website.

**How the models connect.** Space weather (3) drives density (4), density drives drag, drag widens the orbit error (2), and wider orbit error raises collision risk (1). The console shows that chain on one screen when a storm is coming.

**ML engineering that applies to every module:** fixed seeds and one command to retrain, MLflow for runs and a model registry, DVC for data versions, model cards on Hugging Face, ONNX for serving, a drift job that compares the last 30 days of live error with the test error and fails a GitHub check when it drifts, and ablation tables in each model card.

---

## Product layer

The console is built around one decision per conjunction: manoeuvre, monitor, or dismiss, made by a person and logged with a reason. See `product/` for the brief, personas, PRD, metrics tree, evaluation plan and decision log.

---

## Deployment layer (FDE)

What an operator would need before they could use Ashvini on Monday:

| Need | How Ashvini handles it |
|---|---|
| Their own orbits are better than public TLEs | Upload CCSDS OEM ephemeris or OMM. Their objects are screened with their data, everything else with public data |
| They already receive CDMs from Space-Track or a commercial provider | CCSDS CDM (KVN) parser. Ashvini re-scores incoming CDMs with the risk model and keeps the history per event |
| Alerts go where they already work | Webhooks, Slack, email, with per-team thresholds |
| Different teams, different satellites | Organisations, roles (viewer, analyst, admin), API keys, audit log |
| Some cannot use the cloud | `docker compose up` on their own server, Postgres included, no outside calls except the data sources they enable |
| They need to trust it | Every number links to its inputs; every model has a card; every alert can be replayed |
| They need it to keep working | Health endpoint with data age per source, runbook in `docs/runbooks/` |

---

## Hosting (all free tiers)

Vercel for the web app, Render for the API, Neon for Postgres, GitHub Actions for the 4-hourly refresh and CI, Hugging Face for models and a Space for heavy demos, Kaggle and Colab for training.
