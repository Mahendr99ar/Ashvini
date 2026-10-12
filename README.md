# Ashvini

*Every object in Earth orbit, and the few close approaches that matter.*

A live 3D console that screens the public satellite catalogue for close approaches, estimates collision risk with a model trained on ESA's real conjunction data, spots satellites that have changed orbit, and forecasts the space weather that makes all of it less certain.

[![CI](https://github.com/Mahendr99ar/ashvini/actions/workflows/ci.yml/badge.svg)](https://github.com/Mahendr99ar/ashvini/actions/workflows/ci.yml)
[![Refresh data](https://github.com/Mahendr99ar/ashvini/actions/workflows/refresh.yml/badge.svg)](https://github.com/Mahendr99ar/ashvini/actions/workflows/refresh.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-3776AB.svg)](backend)
[![CCSDS CDM and OEM](https://img.shields.io/badge/CCSDS-CDM%20%7C%20OEM-555.svg)](backend/ashvini/ingest/ccsds.py)

> **Ashvini is not an official collision warning service.** It runs on public orbital data, which is less precise than what operators and the Space Force hold, and it says so on every event. It recommends; a person decides, and writes down why.

**Live:** `https://ashvini.vercel.app` *(after deployment, see [DEPLOY.md](DEPLOY.md))*
**API docs:** `https://ashvini-api.onrender.com/docs`
**Train every model:** [Google Colab notebook](ml/notebooks/colab_train_all.ipynb) (or [Kaggle](ml/notebooks/kaggle_train_all.ipynb))

Ashvini is the first of the 27 nakshatras, named for the twin healers of the sky.

---

## What this is

Tens of thousands of tracked objects circle the Earth. When two are predicted to pass close, the satellite's operator gets a warning and has to decide: move the satellite, keep watching, or let it go. Moving costs propellant, which is the satellite's remaining life. Not moving, when it mattered, costs the satellite and fills the orbit with debris for years. ISRO alone made 122 collision avoidance manoeuvres in the 14 years covered by its ISSAR 2024 report, and only a small share of warnings ever end in one.

Ashvini is built around that decision. Three layers, each written to be read on its own:

- **Machine learning.** Ten modules, each trained on a public dataset and each compared with a plain baseline before it is allowed near the product.
- **Product.** An operator console where every event carries a recommendation, the thresholds behind it, its history and its decisions. The product thinking is written down in [`product/`](product).
- **Deployment.** The pieces an operator needs before using it on Monday: their file formats, their alert channels, their thresholds, their servers. See the [deployment playbook](product/fde-playbook.md).

---

## Why not just read the warnings?

Because a warning is one number at one moment, and the number moves.

- **Risk changes as tracking improves.** An event at 10⁻⁵ three days out can end at 10⁻³ or at 10⁻¹². ESA's 2019 competition asked whether a model could predict the final value from early warnings; the plain "latest value carried forward" was hard to beat. Ashvini's risk model is measured against exactly that.
- **Space weather moves everything.** A geomagnetic storm swells the upper atmosphere, low satellites slow down, and yesterday's predictions drift. Ashvini puts the Kp forecast next to the events it affects.
- **Satellites move themselves.** A manoeuvre makes the old orbit, and every prediction based on it, wrong. Ashvini watches element histories for the jump.

---

## How it works

### Stage 1: Load the catalogue

`ingest/celestrak.py` reads CelesTrak's general perturbations data (OMM JSON) for active satellites and the big debris clouds, caches it for two hours as CelesTrak asks, and falls back to the cache if a request fails. Operators can upload their own CCSDS OEM ephemerides; EME2000 states are rotated into SGP4's TEME frame with IAU 1976 precession and the largest nutation terms (`orbits/frames.py`).

### Stage 2: Screen for close approaches

`screening/screen.py` runs three passes, cheapest first: the apogee/perigee filter of Hoots et al. (1984), a 30-second k-d tree sieve whose radius allows for 15.8 km/s head-on closing, and Brent's method on the miss distance around each pass. A test plants three encounters at known times and distances; screening finds each one exactly once, within 2 seconds and 10 metres.

### Stage 3: Estimate the risk

`screening/pc.py` computes the probability of collision with the short-encounter model of Foster and Estes (1992): combined covariance projected onto the encounter plane, integrated over a 20 m disc. TLEs carry no covariance, so `screening/covariance.py` grows one with element age, largest along track, in the range Flohrer et al. (2008) measured. The trained model (`ml/collision`) then predicts where the risk is heading, with a conformal 90% range.

### Stage 4: Recommend, never act

`risk/decision.py` turns Pc, the predicted risk, the miss distance, the time left and the age of the data into manoeuvre, monitor or dismiss, with every reason spelled out. Each organisation sets its own lines (default action 10⁻⁴, monitor 10⁻⁶). A note is required to record a manoeuvre or to disagree with the recommendation.

### Stage 5: Watch for change

`manoeuvres.py` flags steps in semi-major axis that are large in kilometres and large for that object. `ingest/spaceweather.py` reads observed and forecast Kp from NOAA SWPC.

### Stage 6: Tell people where they work

`alerts.py` sends each event once per level to webhooks, Slack or email, and again only if it escalates. A read-only copilot answers questions through four tools and checks every number it says against the data.

---

## Architecture

```
 Public and operator data         Pipeline (GitHub Actions, every 4 h)        Serving (free tier)            People
 ------------------------         ------------------------------------        -------------------            ------
 CelesTrak GP (OMM)       --->    ingest, cache, history                      FastAPI on Render   ---------> 3D console (Vercel)
 NOAA SWPC Kp             --->    screen: filter -> k-d sieve -> Brent        Postgres on Neon               three.js + SGP4 in a Web Worker
 Operator OEM, CDM        --->    Pc, covariance, risk model (ONNX)           SSE live stream                Slack, webhooks, email
                                  recommendation, manoeuvres, alerts          read-only copilot              CCSDS CDM export

 Training (Colab or Kaggle, free GPU): ESA CDMs, OMNI, CHAMP/GRACE, ESA-ADB, SPEED+, SDO  --->  model cards + collision-risk.onnx
```

| Part | Stack |
|---|---|
| Orbits and screening | Python, sgp4 (C extension), NumPy, SciPy k-d trees and Brent's method |
| API | FastAPI, SQLAlchemy 2, Postgres (SQLite in tests), server-sent events |
| Models | LightGBM, PyTorch (Transformer, TCN, autoencoder, MC dropout, CNNs), Stable-Baselines3 PPO, ONNX Runtime |
| Web | TypeScript, Vite, three.js, satellite.js in a Web Worker, no UI framework |
| Ops | GitHub Actions CI and cron, Render Blueprint, Vercel, Docker Compose for on-premises |

---

## The models

Every module writes a model card with its baseline, held-out results for both, and whether it won. The website's Models page reads them.

| Module | Data | Baseline | Models | Metric |
|---|---|---|---|---|
| Collision risk | ESA Collision Avoidance Challenge | Latest CDM risk carried forward | LightGBM, CDM-sequence Transformer, deep ensemble, conformal range | ESA challenge score |
| Kp forecast | NASA OMNI | Persistence, 27-day recurrence | LightGBM, TCN, Chronos zero-shot | RMSE, storm TSS |
| Thermosphere density | CHAMP, GRACE, Swarm | NRLMSISE-00 | MLP with Monte Carlo dropout and conformal calibration | Log-density error, coverage |
| Telemetry anomalies | ESA-ADB | Isolation Forest | Convolutional autoencoder | Event-wise F0.5 |
| Orbit error | Space-Track history | Plain SGP4 | LightGBM on SGP4 residuals | Position error at 1, 3, 7 days |
| Manoeuvres | Space-Track history, injected burns | Robust step threshold | LightGBM classifier | F1 |
| Debris families | CelesTrak debris | Nearest orbit plane | HDBSCAN, LightGBM | ARI, top-1 parent |
| Avoidance policy | ESA CDM sequences | Fixed Pc rule | PPO | Cost per event |
| Spacecraft pose | SPEED+ | Mean pose | Direct-regression CNN | ESA pose score |
| Solar flares | SDO with GOES labels | Persistence | CNN, Surya fine-tune planned | TSS |

**Results so far.** From the first Colab run on real data, held-out sets, numbers exactly as the model cards report them. Modules not listed have not been trained on real data yet.

*Kp forecast* (NASA OMNI hourly; train before 2017, validate 2017 to 2019, test from 2020):

| Horizon | Persistence RMSE | LightGBM RMSE | TCN RMSE | Persistence storm TSS | LightGBM storm TSS, tuned call |
|---|---|---|---|---|---|
| 3 h | 0.864 | **0.759** | 0.781 | 0.506 | **0.808** |
| 6 h | 1.104 | **0.955** | 0.974 | 0.360 | **0.675** |
| 12 h | 1.323 | **1.110** | 1.132 | 0.223 | **0.461** |
| 24 h | 1.510 | **1.229** | 1.260 | 0.121 | **0.333** |

The first run had lower error than persistence but called storms (Kp 5 and above) worse, because regression pulls forecasts toward the mean. This run weights storm hours three times in training and calls a storm when the forecast passes a level chosen on validation data (Kp 3.1 at 3 hours, lower further out), so both error and storm skill beat persistence. One caution: the true skill statistic weighs missed storms and false alarms by rate, and with storms this rare a low call level buys hits with many false alarms. Precision of storm calls is not reported yet and should be before anyone acts on them.

*Debris families* (CelesTrak debris groups with at least 30 pieces; three families):

| Task | Baseline | Model |
|---|---|---|
| Recover families without labels (adjusted Rand index) | | HDBSCAN **0.784** |
| Attribute a piece to its family (top-1) | nearest orbit plane 0.640 | LightGBM **0.991** |

Pieces were split at random, so every family appears in both training and test; with only three well-separated families this is an easy test. A harder one holds out whole break-ups.

*Collision risk* (ESA Collision Avoidance Challenge; held-out 12.5% of events):

| Model | Run 3 | Run 5 | Run 6 | Run 6 missed high-risk | Run 6 false alarms |
|---|---|---|---|---|---|
| Latest risk carried forward (baseline) | 0.691 | 0.691 | 0.691 | 3 | 34 |
| Baseline, high-risk line tuned | | | 0.684 | 3 | 33 |
| **LightGBM, all features** | 9.06 | 0.974 | **0.603** | 3 | 33 |
| **LightGBM, serving features (the one the API runs)** | 8.887 | 0.974 | **0.627** | 3 | 33 |
| Transformer on the CDM sequence | 1.358 | 0.902 | 0.684 | 3 | 33 |
| Deep ensemble | 8.416 | 0.915 | 0.703 | 4 | 33 |

The challenge score is lower-is-better. Run 6 is the first run where a model beats carrying the latest risk forward: the served model scores 0.627 against 0.691, 9% lower. Moving the high-risk line alone gets to 0.684. The rest comes from the model's forecast. Every model in run 6 picked the same line (-5.7), so the high-risk calls are identical, and the gain is in how close the forecast lands on events that are truly high risk: the error there falls from 0.360 to 0.330.

How it got here:

- **Run 3:** weighted high-risk events five times and predicted the clipped risk directly. Forecasts crowded just above -6, and LightGBM raised 1,025 false alarms.
- **Run 5:** predicted the change from the latest risk, dropped the weighting, and chose on validation how much of the change to trust (shrink) and where to draw the high-risk line. The validation split held only about 16 high-risk events, and the line it chose did not hold up on test.
- **Run 6:** chose the shrink and the line on 4-fold out-of-fold predictions over train and validation together, with about six times as many high-risk events.

Read this result with care. The model trusts only 10% of its own predicted change (shrink 0.1). The test set holds only about a dozen truly high-risk events (3 missed plus the hits implied by F2), so one seed is not proof; a seed sweep is the next check. The baseline is a strong one: it is the same carried-forward risk the competition used as its naive baseline, and the competition's write-up (Uriot et al., 2022) reports it was hard to beat. The API serves the model only because the card records that the serving model beat the baseline. Without that, it falls back to the current Pc.

*Avoidance policy* (PPO). Trained and tested only on the 1,866 of 11,731 events whose risk reaches 1e-6, with 374 test episodes. PPO has not beaten the fixed rule; the product does not use it.

| Policy | Runs 5, 6 and 7 mean cost |
|---|---|
| Fixed rule (small burn at the last CDM above 1e-4) | **2.44** |
| PPO | 62.07 |
| Never burn | 62.07 |

In all three runs PPO learned never to burn. Each run tried a different fix:

- **Run 6:** scaled the reward and added an entropy bonus.
- **Run 7:** added the following:
  - **Reward shaping.** Potential-based shaping (Ng, Harada and Russell, 1999), with the potential set to minus the expected collision cost at the current risk. It credits a burn on the step it is made, and the best policy is unchanged; a test checks that.
  - **Scaled observations.** ESA's -30 floor values and same-time CDMs no longer distort the network's inputs.
  - **Checkpoint choice.** It keeps the checkpoint that did best on held-out training events.

On synthetic sequences built to the same cost profile, these changes made PPO match or beat the fixed rule, but the real data did not follow.

The run 7 log shows the likely reason. Validation cost was 1.7082 at every checkpoint. Never burning costs 62 on test against 2.44 for the rule, so nearly all of the cost sits in a handful of events, and the validation slice held none of them. With so few events where a burn matters, PPO sees almost no reward for burning. The next thing to try is sampling those rare events more often during training, with importance weights so the objective stays the same. Until then the fixed rule is the honest answer, and it is what an operator would use anyway.

The synthetic runs in CI only prove that every module trains, evaluates and exports end to end.

---

## How it compares

| | Ashvini | Free Space-Track warnings | Commercial SSA service |
|---|---|---|---|
| Free to run | ✓ | ✓ | ✗ |
| Risk trend predicted, with a range | ✓ | ✗ | depends |
| Space weather beside each event | ✓ | ✗ | depends |
| Reads and writes CCSDS CDM and OEM | ✓ | ✓ | ✓ |
| Decisions and audit trail | ✓ | ✗ | ✓ |
| Its own radar or telescope tracking | ✗ | ✓ | ✓ |
| Official assessment | ✗ | ✓ | ✓ |

---

## Run it

```bash
# API and pipeline
cd backend
pip install -e ".[dev,model]"
python -m ashvini.pipeline              # synthetic catalogue, offline
python -m ashvini.pipeline --source celestrak
uvicorn ashvini.api.main:app --reload   # http://localhost:8000/docs

# Web, in a second terminal
cd web && npm install && npm run dev    # http://localhost:5173

# Models: every module on synthetic data in about thirty seconds
cd ml && pip install -e ".[dev,rl]" && python -m ashvini_ml.train_all --synthetic --quick --out /tmp/models

# Tests
cd backend && pytest -q                 # 40 tests
cd ml && pytest -q                      # 10 tests, including a full synthetic training run
cd web && npm test                      # 7 tests
```

Putting it online for free, step by step: [DEPLOY.md](DEPLOY.md). Setting up an operator: [product/fde-playbook.md](product/fde-playbook.md).

---

## API

| Method | Path | Role | What it does |
|---|---|---|---|
| GET | `/api/health` | | Data age per source, last run, model version |
| GET | `/api/catalogue` | | Element sets for the globe |
| GET | `/api/conjunctions` | | Open events; with a key, only that organisation's, with its thresholds |
| GET | `/api/conjunctions/{id}` | | Detail, history, decisions, lineage |
| GET | `/api/conjunctions/{id}/cdm` | | The event as a CCSDS CDM |
| POST | `/api/conjunctions/{id}/reviews` | analyst | Record a decision (append-only) |
| POST | `/api/upload/cdm` | analyst | Bring in an operator's CDM |
| POST | `/api/upload/oem` | analyst | Operator ephemeris for better screening |
| GET | `/api/spaceweather` | | Observed and forecast Kp |
| GET | `/api/manoeuvres` | | Detected orbit changes |
| POST | `/api/copilot` | | Ask a question; read-only, rate limited |
| GET, POST, PUT | `/api/admin/*` | admin | Channels, watch list, thresholds, audit log |
| GET | `/api/stream` | | Server-sent events |

---

## Repository structure

```
ashvini/
├── backend/ashvini/
│   ├── orbits/          elements, SGP4 propagation, frames, operator ephemerides
│   ├── ingest/          CelesTrak, NOAA SWPC, CCSDS CDM and OEM, synthetic catalogue
│   ├── screening/       filters and k-d sieve, covariance model, Pc
│   ├── risk/            features, ONNX risk model, recommendations
│   ├── api/             FastAPI app
│   ├── pipeline.py      one full run
│   ├── alerts.py        webhooks, Slack, email
│   ├── copilot.py       read-only tools, guards, offline writer
│   └── admin.py         organisations, keys, watch lists, channels
├── ml/ashvini_ml/       collision, kp, density, telemetry, orbit_error, manoeuvre, debris, rl, pose, flare
├── ml/notebooks/        Colab and Kaggle notebooks that train everything
├── web/src/             3D console: scene, propagation worker, panels
├── product/             brief, personas, PRD, metrics, evaluation plan, decision log, deployment playbook
├── docs/                architecture, decision records, runbook
├── render.yaml  docker-compose.yml  DEPLOY.md
└── .github/workflows/   CI and the four-hourly refresh
```

---

## What Ashvini will not do

- Command a spacecraft or plan a burn. It recommends; people decide.
- Present a public-data Pc as an official assessment.
- Publish Space-Track data. Only results derived from it leave the training machine.
- Ship a model that loses to its baseline without saying so.

## Limitations

- TLE accuracy limits everything downstream. Operator ephemerides fix that for their own satellites only.
- The covariance model for TLEs is a rough growth law, not a measured covariance.
- The short-encounter Pc model is wrong for slow, co-orbiting encounters, which are common inside one constellation shell.
- Screening runs every four hours on the free tier, so an element set published just after a run waits up to four hours.
- The free API sleeps after 15 idle minutes; the site shows a saved sample until it wakes.

---

## Credits

Research and data from ESA (Collision Avoidance Challenge, ESA-ADB, Kelvins), NASA (OMNI, SDO, JPL), NOAA SWPC, CelesTrak (T. S. Kelso), Space-Track, Stanford SLAB (SPEED+), IBM Research and NASA (Surya), Amazon Science (Chronos), Google Research (TimesFM), ISRO (ISSAR), and the authors listed on the website's References panel: Uriot et al., Pinto et al., Foster and Estes, Alfano, Hoots et al., Flohrer et al., Vallado et al., Kessler and Cour-Palais, Licata and Mehta, Kotowski et al., Park et al., Kisantal et al., Galvez et al., Schulman et al. Map data from Natural Earth.

## License

MIT. See [LICENSE](LICENSE).

---

*Built by [Mahendra Meena](https://www.linkedin.com/in/mahendra-meena-72047b201/). The warning is a number; the decision is a person's.*
