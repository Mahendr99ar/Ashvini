# Operations runbook

## Is it healthy?

`GET /api/health` shows the age of the newest element set, the newest Kp value and the last run, plus the last run's counts and any error.

| Symptom | Likely cause | What to do |
|---|---|---|
| `newest_elements` over 12 h | Refresh workflow failing | Open Actions, read the failed run. CelesTrak outages fall back to the cached copy; a missing cache means the first run after a long gap failed. |
| `last_run.status` is `failed` | Error in the pipeline | The error text is in `last_run.error`. Rerun with "Run workflow" after fixing. |
| `newest_kp` stale | NOAA SWPC changed or is down | The pipeline logs a warning and keeps going. Events are unaffected. |
| Site says it is showing a saved sample | The free API is asleep | Wait a minute; the page retries every 30 s. |
| Alerts not arriving | Channel misconfigured | `GET /api/admin/audit` and the `deliveries` table show failures with the reason. |

## Rotating keys

Issue a new key with `python -m ashvini.admin create-key "<org>" --role analyst`, hand it over, then set `revoked = true` on the old row in `api_keys`. Keys are stored only as salted hashes, so a lost key cannot be recovered, only replaced.

## Changing thresholds

`PUT /api/admin/thresholds` with an admin key. The change is audited and applies to recommendations immediately; stored events are not rewritten.

## Updating the risk model

Train on Colab (`ml/notebooks/colab_train_all.ipynb`), check that `collision-risk.json` says it beats the baseline, copy `collision-risk.onnx` and `collision-risk.serving.json` into `backend/models/`, set `ASHVINI_RISK_MODEL_PATH`, redeploy. The model version appears on `/api/health` and on every event.
