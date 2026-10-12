# Served models

Put the files from a real training run here:

- `collision-risk.onnx` and `collision-risk.serving.json`: the model the API scores events with.
- `*.json` model cards from any module: the website's Models page lists them.

Then set `ASHVINI_RISK_MODEL_PATH=models/collision-risk.onnx`. Without it the API uses the current Pc,
which is the baseline every model is measured against, and says so on every event.

Do not commit a model trained with `--synthetic`. Those runs only prove the code works.
